#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 17:02:13 2019

@author: paris
"""

import numpy as np
import matplotlib.pyplot as plt

from models import PCA

if __name__ == "__main__":

    data = np.load('/Users/paris/Downloads/UnsupervisedBD/032019_transformation_cscl_0.73.npy')
    
    X = data.reshape((data.shape[0], data.shape[1]*data.shape[2]))
    Z_dim = X.shape[1]
    
    model = PCA(X, Z_dim)

    model.fit()

    Z = model.encode(X)
    X_star = model.decode(Z)

    error = np.linalg.norm(X-X_star,2)/np.linalg.norm(X,2)

    plt.figure(1)
    plt.subplot(2,2,1)
    plt.plot(model.values[:20], 'o')
    plt.xlabel('Dimension index')
    plt.ylabel('$\lambda$')
    plt.subplot(2,2,2)
    plt.plot(model.values[:50], 'o')
    plt.xlabel('Dimension index')
    plt.ylabel('$\lambda$')
    plt.subplot(2,2,3)
    plt.plot(model.values[:100], 'o')
    plt.xlabel('Dimension index')
    plt.ylabel('$\lambda$')
    plt.subplot(2,2,4)
    plt.plot(np.log(model.values[:80]), 'o')
    plt.xlabel('Dimension index')
    plt.ylabel('$\log \lambda$')
    
    np.savetxt('PCA_eigenvalues_new', model.values) 
    np.savetxt('PCA_modes_new', model.vectors)
    

