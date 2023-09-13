from sklearn.cluster import KMeans
import numpy as np
import os
# import cPickle as pickle
import pickle
import importlib
import torch

torch.multiprocessing.set_sharing_strategy('file_system')
# from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
# from scipy.sparse.linalg.eigen.arpack import eigs as largest_eigs
from scipy.linalg import qr
import time
from scipy.sparse import csc_matrix, csr_matrix
from numpy import linalg as LA
import random
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader
from multiprocessing.pool import ThreadPool
import multiprocessing as mp
from tqdm import tqdm
import dataset_network

from config.networks import default as cfg

def solve_envs():
    mod = importlib.import_module(f'config.networks')
    cfg = getattr(mod, 'default')
    print(cfg)
    # update config
    for k, v in cfg.items():
            print(k, '\t', v)
    return cfg

def load_data(cfg):
    config = cfg
    data_ = dataset_network.get_dataset(config)
    if config.dataset == 'icdm_test':
        features, edge_index, node_index = data_.data_tensor()
        return edge_index, features, node_index
    else:
        features, true_clusters, edge_index, node_index = data_.data_tensor()
        return edge_index, features, true_clusters, node_index

if __name__ == '__main__':
    cfg = solve_envs()
    edge_index, features, node_index = load_data(cfg)
    network_data = Data(x=features, edge_index=edge_index, node_index = node_index)
    cluster_data_save_dir = 'cluster_data/'
    cluster_data = ClusterData(network_data, num_parts=10, recursive=False, save_dir=cluster_data_save_dir)
    del node_index
    del edge_index
    data_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=True, num_workers=16)
    del cluster_data
    num_batchs = len(data_loader)
    #available_list = [i for i in range(features.shape[0])]
    #print(f'origin_len: {len(available_list)}')
    node_dict = dict()
    for i, batch in enumerate(data_loader):
        print(batch.order)
        # def get_index(ith):
        #     nonzero_indices = torch.nonzero(torch.all(batch.x[ith] == features, dim=1))
        #     return nonzero_indices.shape[0]
        #     if nonzero_indices.shape[0]>1:
        #         for i_repeat in range(nonzero_indices.shape[0]):
        #             ind_repeat = nonzero_indices[i_repeat].item()
        #             # if ind_repeat in available_list:
        #             #     available_list.remove(ind_repeat)
        #             print(ind_repeat)
        #             return ind_repeat
        #     else:
        #         print(nonzero_indices.item())
        #         return nonzero_indices.item()       
        # fs_list = [inde for inde in range(batch.x.shape[0])]
        # fs_list = fs_list[:1000]

        # with mp.Pool(processes=min(4,len(fs_list))) as p:
        #     result = list(tqdm(p.imap_unordered(get_index, fs_list), total=len(fs_list)))
    #     node_dict[f'batch{i}'] = result
    # print(f'after_len: {len(available_list)}')
    # file_path = "/home/chensiqi/lhtz/ACMin/data/icdm2023_session1_test/processed/node_list.pkl"
    # with open(file_path, "wb") as file_:
    #     pickle.dump(node_dict, file_)
