# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:28:06 2016

@author: ryleyhiga
"""

import numpy as np

def initializeMultinomialClusters(X, T = 30):
    '''
    X = dataset of interest
    T = number of topics
    '''
    N = X.shape[0] # number of documents
    d = X.shape[1] # number of features
    
    #initialize priors
    pi = np.ones(shape = T)/T
    
    #initialize multinomial parameters/cluster centers
    P = np.zeros(shape = (d, T)) # multinomial parameters, rows = features, cols = topics
    init_centers = np.random.randint(0, N-1, T)
    for i, idx in enumerate(init_centers):
        P[:, i] = (X[idx, :] + 1)/(np.sum(X[idx, :]) + d)
        
    return pi, P

def multinomialEStep(X, P, pi, W):
    W = np.dot(X, np.log(P))+np.log(pi)
    maxs = W.max(axis = 1)
    W = (W.T - maxs).T
    W = (W.T - np.log(np.sum(np.exp(W), axis = 1))).T 
    W = np.exp(W)
    return W
        
def multinomialMStep(X, P, pi, W):
    N = X.shape[0]
    d = X.shape[1]
    pi = np.sum(W, axis = 0)/N
    P = np.dot(X.T, W) + 1.0/d 
    
    colsum = np.sum(P, axis = 0)
    P = P/colsum
    return P, pi
    
def multinomialEM(X, T = 30, niters = 100):
    N = X.shape[0]
    
    #intialize the clusters
    pi, P = initializeMultinomialClusters(X, T)
    W = np.ones(shape = (N, T))
    
    #run E steps and M steps iteratively
    for i in range(niters):
        W = multinomialEStep(X, P, pi, W)
        P, pi = multinomialMStep(X, P, pi, W)
        
    classes = np.argmax(W, axis = 1)
    return classes