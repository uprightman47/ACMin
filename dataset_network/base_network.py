import os, re, sys, time, glob, datetime
import numpy as np
import pandas as pd
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils import data
import networkx as nx
class NetworkDataset(data.Dataset):
    def __init__(self, config):
        super(NetworkDataset, self).__init__()
        self.data_total_path = config.data_total_path
class ogbn_arxivDataset(NetworkDataset):
    def __init__(self, config):
        super(ogbn_arxivDataset, self).__init__(config)
        self.dataset_name = 'ogbn_arxiv'
        self.dataset_path = os.path.join(self.data_total_path, self.dataset_name)
        self.read_raw()
    def read_raw(self):
        #obtain X
        feat_df_path = os.path.join(self.dataset_path, 'node_feat.csv')
        feat_df = pd.read_csv(feat_df_path, header=None)
        self.num_features = feat_df.shape[1]
        feat_df = feat_df.to_numpy()
        self.x = feat_df
        self.nodes = [i for i in range(feat_df.shape[0])]

        # obtain edge_index
        # edges_df_path = os.path.join(self.dataset_path, 'edge.csv')
        # edges_df = pd.read_csv(edges_df_path, header=None)
        # edges_list_path = os.path.join(self.dataset_path, 'edge_list.txt')
        # if not os.path.exists(edges_list_path):
        #     edges_df.to_csv(edges_list_path,index=False,sep=' ')
        # edges_df.to_csv(edges_list_path,index=False,sep=' ')
        # self.graph = nx.read_edgelist(edges_list_path, create_using=nx.Graph(), nodetype=int)
        # edges_df = edges_df.T.to_numpy()
        # self.edge_index = edges_df

        #obtain y
        label_df_path = os.path.join(self.dataset_path, 'node_label.csv')
        label_df = pd.read_csv(label_df_path, header=None)
        label_df = label_df.T.to_numpy()[0]
        self.y = label_df
    def data_tensor(self):
        edges_df_path = os.path.join(self.dataset_path, 'edge.csv')
        edges_df = pd.read_csv(edges_df_path, header=None)
        edges_df = edges_df.T.to_numpy()
        return torch.tensor(self.x, dtype = torch.float), torch.tensor(self.y), torch.tensor(edges_df), torch.tensor(self.nodes)
class ogbn_magDataset(NetworkDataset):
    def __init__(self, config):
        super(ogbn_magDataset, self).__init__(config)
        self.dataset_name = 'ogbn_mag'
        self.dataset_path = os.path.join(self.data_total_path, self.dataset_name)
        self.read_raw()
    def read_raw(self):
        #obtain X
        feat_df_path = os.path.join(self.dataset_path, 'node_feat.csv')
        feat_df = pd.read_csv(feat_df_path, header=None)
        self.num_features = feat_df.shape[1]
        feat_df = feat_df.to_numpy()
        self.x = feat_df
        self.nodes = [i for i in range(feat_df.shape[0])]

        # obtain edge_index
        # edges_df_path = os.path.join(self.dataset_path, 'edge.csv')
        # edges_df = pd.read_csv(edges_df_path, header=None)
        # edges_list_path = os.path.join(self.dataset_path, 'edge_list.txt')
        # if not os.path.exists(edges_list_path):
        #     edges_df.to_csv(edges_list_path,index=False,sep=' ')
        # edges_df.to_csv(edges_list_path,index=False,sep=' ')
        # self.graph = nx.read_edgelist(edges_list_path, create_using=nx.Graph(), nodetype=int)
        # edges_df = edges_df.T.to_numpy()
        # self.edge_index = edges_df

        #obtain y
        label_df_path = os.path.join(self.dataset_path, 'node_label.csv')
        label_df = pd.read_csv(label_df_path, header=None)
        label_df = label_df.T.to_numpy()[0]
        self.y = label_df
    def data_tensor(self):
        edges_df_path = os.path.join(self.dataset_path, 'edge.csv')
        edges_df = pd.read_csv(edges_df_path, header=None)
        edges_df = edges_df.T.to_numpy()
        return torch.tensor(self.x, dtype = torch.float), torch.tensor(self.y), torch.tensor(edges_df), torch.tensor(self.nodes)
class icdmDataset(NetworkDataset):
    def __init__(self, config):
        super(icdmDataset, self).__init__(config)
        self.feature_form = config.feature_form
        self.dataset_name = 'icdm2023_session1_test'
        self.dataset_path = os.path.join(self.data_total_path, self.dataset_name)
        self.read_raw()
    def read_raw(self):
        #obtain X
        if self.feature_form == 'origins':
            feat_df_path = os.path.join(self.dataset_path, 'icdm2023_session1_test_node_feat.txt')
        elif self.feature_form == 'representations':
            feat_df_path = os.path.join(self.dataset_path, 'node_features_original_order.txt')
        else:
            raise NotImplementedError('Not support feature form')
        feat_df = pd.read_csv(feat_df_path, sep=',', header=None)
        self.num_features = feat_df.shape[1]
        feat_df = feat_df.to_numpy()
        self.x = feat_df
        self.nodes = torch.arange(feat_df.shape[0])

        # obtain edge_index
        # edges_df_path = os.path.join(self.dataset_path, 'icdm2023_session1_test_edge.txt')
        # edges_df = pd.read_csv(edges_df_path, header=None)
        # edges_list_path = os.path.join(self.dataset_path, 'edge_list.txt')
        # if not os.path.exists(edges_list_path):
        #     edges_df.to_csv(edges_list_path,index=False,sep=' ')
        # edges_df.to_csv(edges_list_path,index=False,sep=' ')
        # self.graph = nx.read_edgelist(edges_list_path, create_using=nx.Graph(), nodetype=int)
        # edges_df = edges_df.T.to_numpy()
        # self.edge_index = edges_df
    def data_tensor(self):
        edges_df_path = os.path.join(self.dataset_path, 'icdm2023_session1_test_edge.txt')
        edges_df = pd.read_csv(edges_df_path, header=None)
        edges_df = edges_df.T.to_numpy()
        return torch.tensor(self.x, dtype = torch.float), torch.tensor(edges_df), self.nodes
