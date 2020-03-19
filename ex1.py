#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for linear regression project
"""

import numpy as np
from matplotlib import pyplot as plt 
from ex1_functions import *

data1=np.loadtxt(fname="ex1data1.txt",delimiter=",")

# Plotting data
print("Plotting data \n")
X=data1[:,0]
y=data1[:,1]
m=len(X) # number of training examples

X=np.reshape(X,[m,1]) #reshaping into vectors
y=np.reshape(y,[m,1])

plt.title("Training data") 
plt.xlabel("City population in 10,000s") 
plt.ylabel("Food truck profit in $10,000") 
plt.plot(X,y,'o') 
plt.show()

wait = input("Program paused. Press enter to continue \n")

X=np.column_stack((np.ones([m,]),data1[:,0])) #add a column of ones to X
theta=np.zeros([2,1]) #initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

print('Testing the cost function ...\n')
#compute and display initial cost
J = computeCost(X, y, theta);
print('theta = [0 ; 0]\nCost computed = {}\n'.format(J))
print('Expected cost value (approx) 32.07\n');

#further testing of the cost function

J=computeCost(X,y,np.reshape(np.array([-1,2]),[2,1]))
print("theta = [-1 ; 2]\nCost computed = {} \n".format(J))
print('Expected cost value (approx) 54.24\n')

wait = input("Program paused. Press enter to continue \n")