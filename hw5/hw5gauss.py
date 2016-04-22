# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 12:01:47 2016

@author: ryleyhiga
"""
import numpy as np

#initial the cluster centers and the prior probabilities. Cluster centers are random    
def intializeClusters(X, C, kmeans = True):
    N = X.shape[0]
    d = X.shape[1]
    pi = np.ones(shape = C)/C
    M = np.zeros(shape = (d, C))
    items = np.random.randint(0, N, C)
    k = 0
    for i in items:
        M[:, k] = X[i]
        k += 1
    
    if (kmeans):
        return kmeansClustering(X, M, pi, niters = 2)
    else:
        return pi, M
    
def kmeansClustering(X, M, pi, niters = 20):
    N = X.shape[0] 
    C = M.shape[1]
    distances = np.ones(shape = (N, C))
    assignments = np.zeros(shape = N)
    for i in range(niters):    
        #assign pts to cluster
        for j in range(C):
            distances[:, j] = np.linalg.norm((X - M[:, j]), axis = 1)
        assignments = distances.argmin(axis = 1)
    
        #recalcuate centers
        for j in range(C):
            idx = np.where(assignments == j)
            count = idx[0].size
            M[:,j] = (X[idx]).sum(axis = 0)/count
            pi[j] = float(count)/N
    return pi, M

def gauss(X, mu):
    '''
    X = flattened dataset
    mu = cluster center
    '''
    distances = np.linalg.norm((X - mu), axis = 1)**2
    return np.exp(-0.5*distances)
    
# perform E step with mixture of Gaussians model
def gaussEStep(X, M, pi, W):
    '''
    X = flattened dataset
    M = mean centers
    pi = priors
    W = soft weights
    '''
    C = M.shape[1] #number of clusters    
    
    # calculate unnormalized weights
    for j in range(C):
        W[:, j] = pi[j]*gauss(X, M[:, j])
    
    #normalize W over columns
    weightUnit = W.sum(axis = 1)
    W = (W.T/weightUnit).T;
    #W = W/np.sum(W, axis = 0)
    return W

# perform M step with mixture of Gaussians model
def gaussMStep(X, M, pi, W):
    '''
    X = flattened dataset
    M = mean centers
    pi = priors
    W = soft weights
    '''
    N = X.shape[0]
    M = np.dot(X.T, W)
    
    #calculate
    clusterWeight = np.sum(W, axis = 0)
    M = M / clusterWeight
    
    # calculate priors
    pi = clusterWeight / N
    
    return M, pi
    
def gaussEM(X, C, niters = 100, kmeans = True):
    N = X.shape[0]
    result = np.ones(shape = X.shape)
    std0 = np.std(X[:, 0])/2
    std1 = np.std(X[:, 1])/2
    std2 = np.std(X[:, 2])/2
    result[:, 0] = X[:, 0]/std0
    result[:, 1] = X[:, 1]/std1
    result[:, 2] = X[:, 2]/std2
    
    #intialize the clusters
    pi, M = intializeClusters(result, C, kmeans)
    
    W = np.ones(shape = (N, C))
    
    #run E steps and M steps iteratively
    #for i in range(niters):
    for i in range(niters):
        W = gaussEStep(result, M, pi, W)
        M, pi = gaussMStep(result, M, pi, W)
        
    classes = np.argmax(W, axis = 1)
    for i in range(C):
        idx = np.where(classes == i)
        result[idx, :] = M[:, i]

    result[:, 0] = result[:, 0]*std0
    result[:, 1] = result[:, 1]*std1
    result[:, 2] = result[:, 2]*std2
    result = result.astype(int)  
    return classes, result