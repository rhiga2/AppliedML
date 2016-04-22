# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 03:47:02 2016


@author: ryleyhiga
"""
from PIL import Image
import numpy as np
from hw5gauss import *

segments = [10, 20, 50]

while True:
    try:
        file = input("Give path of input file such as balloons.jpg: ")
        im = Image.open(file)
        break
    except FileNotFoundError:
        print("File not found.")         
    

nrows = im.size[1]
ncols = im.size[0]
d = 3 #RGB => 3 features
X = np.array(im.getdata(), dtype = float)

C_idx = 3
while (C_idx < 0 or C_idx > 2):  
    C_idx = int(input("Please input 0 for 10 clusters, 1 for 20 clusters, or 2 for 50 clusters: "))
C = segments[C_idx] 

classes, result = gaussEM(X, C, niters = 50, kmeans = False)

out = Image.new("RGB", (ncols, nrows))
for i in range(nrows):
    for j in range(ncols):
        out.putpixel((j, i), tuple(result[i*ncols+j]))
        
while True:
    try:
        outfile = input("Give name of output file such as oceans_segmented30.jpg: ")
        im = out.save(outfile)
        break
    except FileNotFoundError:
        print("File not found.")  
