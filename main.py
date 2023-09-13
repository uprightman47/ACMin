#########################################################################
# File Name: node_cluster.py
# Author: anryyang
# mail: anryyang@gmail.com
# Created Time: Fri 05 Apr 2019 04:33:10 PM
#########################################################################
#!/usr/bin/env/ python

from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import argparse
import os
import importlib
# import cPickle as pickle
import _pickle as cPickle
import networkx as nx
from scipy.sparse import identity
from munkres import Munkres
from sklearn import preprocessing
from sklearn.decomposition import NMF
import heapq
from spectral import discretize
# from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
# from scipy.sparse.linalg.eigen.arpack import eigs as largest_eigs
from scipy.linalg import qr
from scipy.linalg import orth
from scipy.sparse.csgraph import laplacian
import time
import sklearn
from sklearn.linear_model import SGDRegressor
from scipy.sparse import csc_matrix, csr_matrix
from numpy import linalg as LA
import operator
import random


import dataset_network
from config.networks import default as cfg
from utils.clustering import ACMin


print(sklearn.__version__)

def solve_envs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg_path', type=str, help='config path')
    FLAGS = parser.parse_args()
    # ---------------------------------------------------------------------------- #
    # solve env & cfg
    # ---------------------------------------------------------------------------- #
    assert FLAGS.cfg_path is not None
    cfg = [i for i in FLAGS.cfg_path.split('.') if i != 'config']
    mod = importlib.import_module(f'config.{cfg[0]}')
    cfg = getattr(mod, cfg[1])
    print(cfg)
    # update config
    for arg in ['dataset', 'cluster_params', 'num_cluster']:
        if getattr(FLAGS, arg, None) is not None:
            setattr(cfg, arg, getattr(FLAGS, arg))
    for k, v in cfg.items():
            print(k, '\t', v)
    return cfg




def load_data(cfg):
    data_ = dataset_network.get_dataset(cfg)
    if cfg.dataset == 'icdm_test':
        features = data_.x
        graph = data_.graph
        return graph, features
    else:
        features = data_.x
        true_clusters = data_.y
        graph = data_.graph
        return graph, features, true_clusters
def obtain_cluster_results(cfg):
    if cfg.dataset == 'icdm_test':
        graph, feats = load_data(cfg)
    else:
        graph, feats, true_clusters = load_data(cfg)
    if cfg.num_cluster:
        num_cluster = cfg.num_cluster
    else:
        num_cluster = len(np.unique(true_clusters))
    if cfg.mode == 'cluster':
        predict_clusters = cluster(graph, feats, num_cluster, **cfg.cluster_params)
        K = len(set(predict_clusters))
        print("-------------------------------")
        file_path = os.path.join(cfg.cluster_result_path, "sc."+cfg.dataset+"."+str(K)+".cluster.txt")
        with open(file_path, "w") as fout:
            for i in range(len(predict_clusters)):
                fout.write(str(predict_clusters[i])+"\n")
        print("---------------finish saving----------------")
        if cfg.dataset != 'icdm_test':
            cm = clustering_metrics(true_clusters, predict_clusters)
            print("acc: %f\t nmi: %f\t adjscore: %f\t ari:%f"%cm.evaluationClusterModelFromLabel())
    # elif cfg.mode == 'eval_metrics':
    #     file_path = os.path.join(cfg.cluster_result_path, "sc."+cfg.dataset+"."+str(K)+".cluster.txt")

if __name__ == '__main__':
    cfg = solve_envs()
    ACMin(cfg)

