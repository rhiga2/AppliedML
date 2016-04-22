# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 20:29:28 2016

@author: ryleyhiga
"""
from hw5multinomial import *
import numpy as np
import matplotlib.pyplot as plt

#import data and word list
#import dataset store as nd array
with open('docword.nips.txt') as nips_file:
    nips_strlist = nips_file.read().split('\n')
    
D = int(nips_strlist[0]) #number of documents
W = int(nips_strlist[1]) #number of words in the vocabulary
N = int(nips_strlist[2]) #number of total words in dataset
T = 30 #number of clusters
nips_data = np.zeros(shape = (D,W))
del nips_strlist[0]
del nips_strlist[0]
del nips_strlist[0]
del nips_strlist[-1]

for row in nips_strlist:
    row_list = row.split(' ')
    i = int(row_list[0])-1 #0-indexing
    j = int(row_list[1])-1
    k = int(row_list[2])
    nips_data[i][j] = k

#import wordlist
with open('vocab.nips.txt') as nipswords_file:
    wordlist = nipswords_file.read().split('\n')
    
classes  = multinomialEM(nips_data, T = T, niters = 20)

clusterMatrix = np.zeros(shape = (T, W))
proportions = np.zeros(shape = T)
for i, cluster in enumerate(classes):
    clusterMatrix[cluster, :] += nips_data[i]
    proportions[cluster] += 1

#plot proportion of documents per cluster
proportions = proportions/(proportions.sum())
plt.figure()
plt.title("Proportion of Documents in Topic")
plt.xlabel("Topic")
plt.ylabel("Proportion")
plt.plot(range(1, 31), proportions)

comWords = []
for i in range(T):
    comWordsPerTopic = []
    for j in range(10):
        k = np.argmax(clusterMatrix[i, :])
        comWordsPerTopic.append(wordlist[k])
        clusterMatrix[i, k] = 0
    comWords.append(comWordsPerTopic)
print(comWords)