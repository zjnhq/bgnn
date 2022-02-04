import itertools
import time
import numpy as np
import torch
import pandas as pd
from catboost import Pool, CatBoostClassifier, CatBoostRegressor, sum_models
from .GNN import GNNModelDGL, GATDGL
from .Base import BaseModel

# from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from tqdm import tqdm
from collections import defaultdict as ddict
from pdb import set_trace
class BGNN(BaseModel):
    def __init__(self,
                 task='regression', iter_per_epoch = 10, lr=0.01, hidden_dim=64, dropout=0.,
                 only_gbdt=False, train_non_gbdt=False,
                 name='gat', use_leaderboard=False, depth=6, gbdt_lr=0.1, use_graphgbm = 0):
        super(BaseModel, self).__init__()
        self.learning_rate = lr
        self.hidden_dim = hidden_dim
        self.task = task
        self.dropout = dropout
        self.only_gbdt = only_gbdt
        self.train_residual = train_non_gbdt
        self.name = name
        self.use_leaderboard = use_leaderboard
        self.iter_per_epoch = iter_per_epoch
        self.depth = depth
        self.lang = 'dgl'
        self.gbdt_lr = gbdt_lr
        self.use_graphgbm = (use_graphgbm != 0)
        self.graphgbm_beta = use_graphgbm
        if self.use_graphgbm:
            self.iter_per_epoch = int(self.iter_per_epoch / 2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def __name__(self):
        return 'BGNN'

    def init_gbdt_model(self, num_epochs, epoch):
        if self.task == 'regression':
            catboost_model_obj = CatBoostRegressor
            catboost_loss_fn = 'RMSE' #''RMSEWithUncertainty'
        else:
            if epoch == 0:
                catboost_model_obj = CatBoostClassifier
                catboost_loss_fn = 'MultiClass'
            else:
                catboost_model_obj = CatBoostRegressor
                catboost_loss_fn = 'MultiRMSE'

        return catboost_model_obj(iterations=num_epochs,
                                  depth=self.depth,
                                  learning_rate=self.gbdt_lr,
                                  loss_function=catboost_loss_fn,
                                  random_seed=0,
                                  nan_mode='Min')

    def fit_gbdt(self, pool, trees_per_epoch, epoch):
        gbdt_model = self.init_gbdt_model(trees_per_epoch, epoch)
        gbdt_model.fit(pool, verbose=False)
        return gbdt_model

    def init_gnn_model(self):
        # set_trace()
        if self.use_leaderboard:
            self.model = GATDGL(in_feats=self.in_dim, n_classes=self.out_dim).to(self.device)
        else:
            self.model = GNNModelDGL(in_dim=self.in_dim,
                                     hidden_dim=self.hidden_dim,
                                     out_dim=self.out_dim,
                                     name=self.name,
                                     dropout=self.dropout).to(self.device)

    def append_gbdt_model(self, new_gbdt_model, weights):
        if self.gbdt_model is None:
            return new_gbdt_model
        return sum_models([self.gbdt_model, new_gbdt_model], weights=weights)

    def train_gbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha): 
        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_gbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch, epoch)
        if epoch == 0 and self.task=='classification':
            self.base_gbdt = epoch_gbdt_model
        else:
            self.gbdt_model = self.append_gbdt_model(epoch_gbdt_model, weights=[1, gbdt_alpha])

    def train_graphgbdt(self, gbdt_X_train, gbdt_y_train, cat_features, epoch,
                   gbdt_trees_per_epoch, gbdt_alpha): 
        
        pool = Pool(gbdt_X_train, gbdt_y_train, cat_features=cat_features)
        epoch_graphgbdt_model = self.fit_gbdt(pool, gbdt_trees_per_epoch, epoch)
        if self.graphgbdt_model is None:
            self.graphgbdt_model = epoch_graphgbdt_model
        if epoch == 0 and self.task=='classification':
            self.base_graphgbdt = epoch_graphgbdt_model
        else:
            self.graphgbdt_model = sum_models([self.graphgbdt_model, epoch_graphgbdt_model], weights=[1, gbdt_alpha])

    def update_binode_prediction(self, cur_prediction, edge_X, edge_starts):
        
        if(self.graphgbdt_model is None):
            return 0.0
        if (edge_X.shape[0]!=edge_starts.shape[0]):
            print(" \n bi-node features and index dimension mismatch")
            return 0 
        if self.task == 'regression':
            graph_predictions=self.graphgbdt_model.predict(edge_X)
            graph_predictions = np.expand_dims(graph_predictions, axis=1)
            #dim: (num_edges)
        else:
            graph_predictions = self.base_graphgbdt.predict_proba(edge_X) 
            if self.gbdt_model is not None:
                graph_predictions += self.graphgbdt_model.predict(edge_X)
            graph_predictions += self.edge_ends_y_train
            #boosting machine on bi-node features predicts y[edge_start] - y[edge_end]

        predictions_add_by_graph = np.zeros(cur_prediction.shape)
        predictions_added_num_times = np.ones(cur_prediction.shape[0]) 
        # make a mixture of node prediction and bi-node prediction, by a factor of beta=0.8
        # for each node, it could have multiple edges contributing bi-node predictions, 
        # so average over them
        for i in range(graph_predictions.shape[0]):
            node = edge_starts[i]
            predictions_add_by_graph[node] += graph_predictions[i]
            predictions_added_num_times[node] += 1

        # predictions_add_by_graph[predictions_added_num_times==0] = cur_prediction[predictions_added_num_times==0]
        # predictions_added_num_times[predictions_added_num_times==0] = 1
        predictions_add_by_graph = np.divide(predictions_add_by_graph,predictions_added_num_times[:,np.newaxis])
        return predictions_add_by_graph * self.graphgbm_beta

    def update_node_features(self, node_features, X, encoded_X):
        # set_trace()
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
            # predictions = self.gbdt_model.virtual_ensembles_predict(X,
            #                                                         virtual_ensembles_count=5,
            #                                                         prediction_type='TotalUncertainty')
        else:
            predictions = self.base_gbdt.predict_proba(X)
            # predictions = self.base_gbdt.predict(X, prediction_type='RawFormulaVal')
            if self.gbdt_model is not None:
                predictions_after_one = self.gbdt_model.predict(X)
                predictions += predictions_after_one
        if self.use_graphgbm:
            if self.edge_starts.max()>X.shape[0]:
                print("update function cannot apply here")
            else:
                graph_predictions = self.update_binode_prediction(predictions, self.edge_features, self.edge_starts)
                predictions = predictions+ graph_predictions
        if not self.only_gbdt:
            if self.train_residual:
                predictions = np.append(node_features.detach().cpu().data[:, :-self.out_dim], predictions,
                                        axis=1)  # append updated X to prediction
            else:
                predictions = np.append(encoded_X, predictions, axis=1)  # append X to prediction

        predictions = torch.from_numpy(predictions).to(self.device)

        node_features.data = predictions.float().data

    def update_gbdt_targets(self, node_features, node_features_before, train_mask):
        updated_target = (node_features - node_features_before).detach().cpu().numpy()[train_mask, -self.out_dim:]
        if self.use_graphgbm:
            self.edge_ends_y_train_curent_estimate = (node_features[self.edge_starts_train] - node_features_before[self.edge_ends_train]).detach().cpu().numpy()[:, -self.out_dim:]
        return updated_target

    def init_node_features(self, X):
        node_features = torch.empty(X.shape[0], self.in_dim, requires_grad=True, device=self.device)
        if not self.only_gbdt:
            node_features.data[:, :-self.out_dim] = torch.from_numpy(X.to_numpy(copy=True))
        return node_features

    def init_node_parameters(self, num_nodes):
        return torch.empty(num_nodes, self.out_dim, requires_grad=True, device=self.device)

    def init_optimizer2(self, node_parameters, learning_rate):
        params = [self.model.parameters(), [node_parameters]]
        return torch.optim.Adam(itertools.chain(*params), lr=learning_rate)

    def update_node_features2(self, node_parameters, X):
        if self.task == 'regression':
            predictions = np.expand_dims(self.gbdt_model.predict(X), axis=1)
        else:
            predictions = self.base_gbdt.predict_proba(X)
            if self.gbdt_model is not None:
                predictions += self.gbdt_model.predict(X)

        predictions = torch.from_numpy(predictions).to(self.device)
        node_parameters.data = predictions.float().data

    def build_graphgbdt_features(self, networkx_graph, X, y, train_mask_binary, test_mask_binary):
        edges = np.asarray(list(networkx_graph.edges))
        self.edge_starts = edges[:,0]
        self.edge_ends = edges[:,1]
        edge_difference = np.array(X.iloc[self.edge_starts,:]) - np.array( X.iloc[self.edge_ends,:])
        edge_common = np.array(X.iloc[self.edge_starts,:]) - np.array( X.iloc[self.edge_ends,:])
        self.edge_features = np.stack([X.iloc[self.edge_starts,:], X.iloc[self.edge_ends,:], edge_difference, edge_common],1)
        self.edge_features = self.edge_features.reshape(self.edge_features.shape[0], -1)
        #bi-node features are concat features of two nodes
        self.edge_X_train = self.edge_features.copy()
        self.edge_starts_train = self.edge_starts.copy()
        self.edge_ends_train = self.edge_ends.copy()
        
        self.edge_starts_test = self.edge_starts.copy()
        self.edge_ends_test = self.edge_ends.copy()
        
        j = 0
        k = 0
        for index in range(len(self.edge_starts)):
            node_ends = self.edge_ends[index] 
            node_starts = self.edge_starts[index]
            if train_mask_binary[node_starts] and train_mask_binary[node_ends]: # as long as one node is trained
                self.edge_X_train[j] =self.edge_features[index]
                self.edge_starts_train[j] = node_starts
                self.edge_ends_train[j] = node_ends
                j +=1
            if test_mask_binary[node_starts]:
                self.edge_starts_test[k] = node_starts
                self.edge_ends_test[k] = node_ends
                k+=1
            elif test_mask_binary[node_ends]:
                self.edge_starts_test[k] = node_ends 
                self.edge_ends_test[k] = node_starts
                k+=1

        self.edge_X_train = self.edge_X_train[:j]
        self.edge_starts_train = self.edge_starts_train[:j]
        self.edge_ends_train = self.edge_ends_train[:j]
        
        self.edge_starts_test = self.edge_starts_test[:k]
        self.edge_ends_test = self.edge_ends_test[:k]
        edge_difference = np.array(X.iloc[self.edge_starts_test,:]) - np.array( X.iloc[self.edge_ends_test,:])
        edge_common = np.array(X.iloc[self.edge_starts_test,:]) - np.array( X.iloc[self.edge_ends_test,:])
        self.edge_X_test = np.stack([X.iloc[self.edge_starts_test,:], X.iloc[self.edge_ends_test,:], edge_difference, edge_common],1)
        self.edge_X_test = self.edge_X_test.reshape(self.edge_X_test.shape[0], -1)

        self.edge_X_train = pd.DataFrame(self.edge_X_train)
        self.edge_ends_y_train = np.array(y.iloc[self.edge_ends_train])
        self.edge_ends_y_train_curent_estimate = np.zeros([self.edge_ends_train.shape[0],1])
        self.edge_starts_y_train = np.array(y.iloc[self.edge_starts_train,:])

        self.edge_X_test = pd.DataFrame(self.edge_X_test)
        self.edge_ends_test = np.array(y.iloc[self.edge_ends_test])
        self.edge_ends_y_test_curent_estimate = np.zeros([self.edge_ends_test.shape[0],1])
        print('using training %d edges that has both nodes labeled', j)
        print('using testing %d edges that one node labeled', k)

    def fit(self, networkx_graph, X, y, train_mask, val_mask, test_mask, cat_features,
            num_epochs, patience, logging_epochs=1, loss_fn=None, metric_name='loss',
            normalize_features=True, replace_na=True,
            ):

        # initialize for early stopping and metrics
        if metric_name in ['r2', 'accuracy']:
            best_metric = [np.float('-inf')] * 3  # for train/val/test
        else:
            best_metric = [np.float('inf')] * 3  # for train/val/test
        best_val_epoch = 0
        epochs_since_last_best_metric = 0
        metrics = ddict(list)
        if cat_features is None:
            cat_features = []

        if self.task == 'regression':
            self.out_dim = y.shape[1]
        elif self.task == 'classification':
            self.out_dim = len(set(y.iloc[test_mask, 0]))
        # self.in_dim = X.shape[1] if not self.only_gbdt else 0
        # self.in_dim += 3 if uncertainty else 1
        self.in_dim = self.out_dim + X.shape[1] if not self.only_gbdt else self.out_dim

        self.init_gnn_model()

        gbdt_X_train = X.iloc[train_mask]
        gbdt_y_train = y.iloc[train_mask]
        gbdt_alpha = 1
        self.gbdt_model = None
        self.graphgbdt_model = None

        encoded_X = X.copy()
        if not self.only_gbdt:
            if len(cat_features):
                encoded_X = self.encode_cat_features(encoded_X, y, cat_features, train_mask, val_mask, test_mask)
            if normalize_features:
                encoded_X = self.normalize_features(encoded_X, train_mask, val_mask, test_mask)
            if replace_na:
                encoded_X = self.replace_na(encoded_X, train_mask)

        if self.use_graphgbm and (not hasattr(self, 'edge_X_train')):
            y_in_pandas = y.copy()
            train_mask_binary = np.zeros(max(max(max(train_mask), max(test_mask)), max(val_mask))+1,dtype=int)
            train_mask_binary[train_mask]=1
            test_mask_binary = np.zeros(max(max(max(train_mask), max(test_mask)), max(val_mask))+1,dtype=int)
            test_mask_binary[test_mask]=1
            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = test_mask
            self.build_graphgbdt_features(networkx_graph, X, y_in_pandas, train_mask_binary, test_mask_binary)
            if len(cat_features):
                self.edge_X_cat_train = np.stack([cat_features[self.edge_starts,:], X.iloc[self.edge_ends,:]],1)
                self.edge_X_cat_train = self.edge_X_cat_train.reshape(self.edge_X_cat_train.shape[0], -1)
            else:
                self.edge_X_cat_train = [0]*self.edge_X_train.shape[0]

        node_features = self.init_node_features(encoded_X)
        optimizer = self.init_optimizer(node_features, optimize_node_features=True, learning_rate=self.learning_rate)

        y, = self.pandas_to_torch(y)
        self.y = y
        if self.lang == 'dgl':
            graph = self.networkx_to_torch(networkx_graph)
        elif self.lang == 'pyg':
            graph = self.networkx_to_torch2(networkx_graph)
        
        self.graph = graph

        pbar = tqdm(range(num_epochs))
        for epoch in pbar:
            start2epoch = time.time()
            # gbdt part
            self.train_gbdt(gbdt_X_train, gbdt_y_train, cat_features, epoch,
                            self.iter_per_epoch, gbdt_alpha)

            self.update_node_features(node_features, X, encoded_X)
            node_features_before = node_features.clone()
            model_in=(graph, node_features)
            loss = self.train_and_evaluate(model_in, y, train_mask, val_mask, test_mask,
                                           optimizer, metrics, self.iter_per_epoch)
            gbdt_y_train = self.update_gbdt_targets(node_features, node_features_before, train_mask)

            if self.use_graphgbm:
                self.graphgbm_y = pd.DataFrame( self.edge_ends_y_train_curent_estimate, columns=['y'])
                self.train_graphgbdt(self.edge_X_train, self.graphgbm_y, [], epoch,
                            self.iter_per_epoch, gbdt_alpha)

            self.log_epoch(pbar, metrics, epoch, loss, time.time() - start2epoch, logging_epochs,
                           metric_name=metric_name)
            # check early stopping
            best_metric, best_val_epoch, epochs_since_last_best_metric = \
                self.update_early_stopping(metrics, epoch, best_metric, best_val_epoch, epochs_since_last_best_metric,
                                           metric_name, lower_better=(metric_name not in ['r2', 'accuracy']))
            if patience and epochs_since_last_best_metric > patience:
                break
            if np.isclose(gbdt_y_train.sum(), 0.):
                print('Nodes do not change anymore. Stopping...')
                break

        if loss_fn:
            self.save_metrics(metrics, loss_fn)

        print('Best {} at iteration {}: {:.3f}/{:.3f}/{:.3f}'.format(metric_name, best_val_epoch, *best_metric))
        return metrics

    def predict(self, graph, X, y, test_mask):
        set_trace()
        node_features = torch.empty(X.shape[0], self.in_dim).to(self.device)
        self.update_node_features(node_features, X, X)
        # self.update_binode_prediction()
        return self.evaluate_model((graph, node_features), y, test_mask)