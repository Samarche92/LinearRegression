#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main file for linear regression project
"""

import numpy as np
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from ex1_functions import *

data1=np.loadtxt(fname="ex1data1.txt",delimiter=",")

# Plotting data
print("*** Plotting data ***\n")
X=data1[:,0]
y=data1[:,1]
m=len(X) # number of training examples

X=np.reshape(X,[m,1]) #reshaping into vectors
y=np.reshape(y,[m,1])

plt.figure()
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

print('*** Testing the cost function ***\n')
#compute and display initial cost
J = computeCost(X, y, theta);
print('theta = [0 ; 0]\nCost computed = {}\n'.format(J))
print('Expected cost value (approx) 32.07\n');

#further testing of the cost function

J=computeCost(X,y,np.reshape(np.array([-1,2]),[2,1]))
print("theta = [-1 ; 2]\nCost computed = {} \n".format(J))
print('Expected cost value (approx) 54.24\n')

wait = input("Program paused. Press enter to continue \n")

print("*** Running gradient descent ***")

theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n{}\n'.format(theta))

print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n');

# Plot the linear fit
plt.plot(X[:,1], np.matmul(X,theta))
plt.legend(['Training data','Linear regression'])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 =np.matmul(np.array([[1, 3.5]]),theta)
print('For population = 35,000, we predict a profit of {}\n'
      .format(predict1*10000))

predict2 =np.matmul(np.array([[1, 7]]),theta)
print('For population = 70,000, we predict a profit of {}\n'
      .format(predict2*10000))

wait = input("Program paused. Press enter to continue \n")

# Visualizing J
print('*** Visualizing J(theta_0, theta_1) ***\n')

# Grid over which we will compute J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

# Fill out J_vals
for (i,t0) in enumerate(theta0_vals):
    for (j,t1) in enumerate(theta1_vals):
        t=np.array([[t0],[t1]])
        J_vals[i,j]=computeCost(X,y,t)
        
theta0_vals,theta1_vals = np.meshgrid(theta0_vals,theta1_vals)

# surface plot       
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals,np.transpose(J_vals),cmap='viridis')
plt.xlabel("theta_0") 
plt.ylabel("theta_1") 
plt.show()

# Contour plot
fig = plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals,np.transpose(J_vals), np.logspace(-2, 3, 20))
plt.xlabel("\theta_0") 
plt.ylabel("\theta_1") 
plt.plot(theta[0], theta[1], 'xr', markersize=12,linewidth=2)
plt.show()
