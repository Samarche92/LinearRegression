#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module containing useful functions
"""

import numpy as np

def computeCost(X,y,theta):
    m=len(X)
    h=np.matmul(X,theta)
    J=np.linalg.norm(h-y)
    J*=J/(2*m)

    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m=len(X)
    
    for it in range(iterations):
        h=np.matmul(X,theta)
        grad=np.matmul(np.transpose(X),h-y)
        theta=theta-alpha*grad/m
        
    return theta
    