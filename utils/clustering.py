from sklearn.cluster import KMeans
import numpy as np
# import cPickle as pickle
import _pickle as cPickle
import networkx as nx
from scipy.sparse import identity
from sklearn import preprocessing
import heapq
from spectral import discretize
# from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
# from scipy.sparse.linalg.eigen.arpack import eigs as largest_eigs
from scipy.linalg import qr
import time
from scipy.sparse import csc_matrix, csr_matrix
from numpy import linalg as LA
import random

def si_eig(P, X, alpha, beta, k, a):
    t = 500
    q, _ = qr(a, mode='economic')
    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    X = X/xsum
    for i in range(t):
        z = (1-alpha-beta)*P.dot(q)+ (beta)*X.dot(XT.dot(q))
        p = q
        q, _ = qr(z, mode='economic')
        if np.linalg.norm(p-q, ord=1)<0.01:
            print("converged")
            break
    
    return q


def base_cluster(graph, X, num_cluster):
    print("attributed transition matrix constrcution...")
    adj = nx.adjacency_matrix(graph)
    P = preprocessing.normalize(adj, norm='l1', axis=1)
    n = P.shape[0]
    print(P.shape)

    start_time = time.time()
    alpha=0.2
    beta=0.35
    XX = X.dot(X.T)
    XX = preprocessing.normalize(XX, norm='l1', axis=1)
    PP = (1-beta)*P + beta*XX
    I = identity(n) 
    S = I
    t = 5 #int(1.0/alpha)
    for i in range(t):
        S = (1-alpha)*PP.dot(S)+I

    S = alpha*S
    q = np.zeros(shape=(n,num_cluster))

    predict_clusters = n*[1]
    lls = [i for i in range(num_cluster)]
    for i in range(n):
        ll = random.choice(lls)
        predict_clusters[i] = ll

    M = csc_matrix((np.ones(len(predict_clusters)), (np.arange(0, n), predict_clusters)),shape=(n,num_cluster+1))
    M = M.todense()

    Mss = np.sqrt(M.sum(axis=0))
    Mss[Mss==0]=1
    q = M*1.0/Mss

    largest_evc = np.ones(shape = (n,1))*(1.0/np.sqrt(n*1.0))
    q = np.hstack([largest_evc,q])

    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    xsum = csr_matrix(1.0/xsum)
    X = X.multiply(xsum)
    print(type(X), X.shape)

    predict_clusters = np.asarray(predict_clusters,dtype=np.int)
    print(q.shape)

    epsilon_f = 0.005
    tmax = 200
    err = 1
    for i in range(tmax):
        z = S.dot(q)
        q_prev = q
        q, _ = qr(z, mode='economic')

        err = LA.norm(q-q_prev)/LA.norm(q)
        if err <= epsilon_f:
            break
        
        if i==tmax-1:
            evecs_large_sparse = q
            evecs_large_sparse = evecs_large_sparse[:,1:num_cluster+1]

            kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_jobs=-1, algorithm='full', init='random', n_init=1, max_iter=50).fit(evecs_large_sparse)
            predict_clusters = kmeans.predict(evecs_large_sparse)


    time_elapsed = time.time() - start_time
    print("%f seconds are taken to train"%time_elapsed)

    return predict_clusters 

def get_ac(P, X, XT, y, alpha, beta, t):
    n = X.shape[0]
    num_cluster = y.max()+1-y.min()
    if(y.min()>0):
        y = y-y.min()
        
    print(n, len(y), num_cluster)
    
    vectors_discrete = csc_matrix((np.ones(len(y)), (np.arange(0, n), y)), shape=(n, num_cluster)).toarray()
    vectors_f = vectors_discrete
    vectors_fs = np.sqrt(vectors_f.sum(axis=0))
    vectors_fs[vectors_fs==0]=1
    vectors_f = vectors_f*1.0/vectors_fs
    q_prime = vectors_f
    h = q_prime
    for tt in range(t):
        h = (1-alpha)*((1-beta)*P.dot(h)+ (beta)*X.dot(XT.dot(h))) +q_prime
    h = alpha*h
    h = q_prime-h
    
    conductance_cur = 0
    for k in range(num_cluster):
        conductance_cur = conductance_cur + (q_prime[:,k].T).dot(h[:,k])#[0,0]
    
    return conductance_cur/num_cluster


def cluster(graph, X, num_cluster, alpha, beta, t, tmax, ri, print_batch):
    print("attributed transition matrix constrcution...")
    adj = nx.adjacency_matrix(graph)
    P = preprocessing.normalize(adj, norm='l1', axis=1)
    n = P.shape[0]
    print(P.shape)
    X = csr_matrix(X)
    epsilon_r = 6*n*np.log(n*1.0)/X.getnnz()
    print("epsilon_r threshold:", epsilon_r)

    degrees = dict(graph.degree())
    topk_deg_nodes = heapq.nlargest(5*t*num_cluster, degrees, key=degrees.get)
    PC = P[:,topk_deg_nodes]
    M = PC
    for i in range(t-1):
        M = (1-alpha)*P.dot(M)+PC
    class_evdsum = M.sum(axis=0).flatten().tolist()
    newcandidates = np.argpartition(class_evdsum, -num_cluster)[-num_cluster:]
    M = M[:,newcandidates]    
    labels = np.argmax(M, axis=1).flatten().tolist()[0]
    labels = np.asarray(labels,dtype=np.int)
    
    # random initialization
    if ri is True:
        lls = np.unique(labels)
        for i in range(n):
            ll = random.choice(lls)
            labels[i] = ll
        
    M = csc_matrix((np.ones(len(labels)), (np.arange(0, M.shape[0]), labels)),shape=(M.shape))
    M = M.todense()

    start_time = time.time()
    
    print("eigen decomposition...")

    Mss = np.sqrt(M.sum(axis=0))
    Mss[Mss==0]=1
    q = M*1.0/Mss

    largest_evc = np.ones(shape = (n,1))*(1.0/np.sqrt(n*1.0))
    q = np.hstack([largest_evc,q])

    XT = X.T
    xsum = X.dot(XT.sum(axis=1))
    xsum[xsum==0]=1
    xsum = csr_matrix(1.0/xsum)
    X = X.multiply(xsum)
    print(type(X), X.shape)
    predict_clusters_best=labels
    iter_best = 0
    conductance_best=100
    conductance_best_acc = [0]*3
    acc_best = [0]*3
    acc_best_iter = 0
    acc_best_conductance = 0
    epsilon_f = 0.005
    err = 1
    for i in range(tmax):
        z = (1-beta)*P.dot(q)+ (beta)*X.dot(XT.dot(q))
        q_prev = q
        q, _ = qr(z, mode='economic')
        
        err = LA.norm(q-q_prev)/LA.norm(q)

        if (i+1)%print_batch==0:
            evecs_large_sparse = q
            evecs_large_sparse = evecs_large_sparse[:,1:num_cluster+1]
            predict_clusters, q_prime = discretize(evecs_large_sparse)
            
            conductance_cur = 0
            h = q_prime
            for tt in range(1):
                h = (1-alpha)*((1-beta)*P.dot(h)+ (beta)*X.dot(XT.dot(h))) +q_prime
            h = alpha*h
            h = q_prime-h
            
            for k in range(num_cluster):
                conductance_cur = conductance_cur + (q_prime[:,k].T).dot(h[:,k])#[0,0]
            conductance_cur=conductance_cur/num_cluster
                
            if conductance_cur<conductance_best:
                conductance_best = conductance_cur
                predict_clusters_best = predict_clusters
                iter_best = i
            if i == print_batch-1:
                time_iter_start = start_time
            time_iter_end = time.time()
            time_iter_interval = time_iter_end - time_iter_start
            time_iter_start = time_iter_end        
            print(f'iter: {i}, error: {err}, conductance_cur: {conductance_cur}, time: {time_iter_interval}')

        if err <= epsilon_f:
            break
    
    if tmax==0:
        evecs_large_sparse = q
        evecs_large_sparse = evecs_large_sparse[:,1:num_cluster+1]
        predict_clusters, q_prime = discretize(evecs_large_sparse)
        predict_clusters_best = predict_clusters
    
    time_elapsed = time.time() - start_time
    print("%f seconds are taken to train"%time_elapsed)
    print(np.unique(predict_clusters_best))
    print("best iter: %d, best condutance: %f, acc: %f, %f, %f"%(iter_best, conductance_best, conductance_best_acc[0], conductance_best_acc[1], conductance_best_acc[2]))
    return predict_clusters_best
    