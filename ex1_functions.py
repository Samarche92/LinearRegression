#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
module containing useful functions
"""

import numpy as np

def computeCost(X,y,theta):
    m=len(X)
    h=np.matmul(X,theta)
    J=np.linalg.norm(h-y)**2
    J/=(2*m)
    return J
    