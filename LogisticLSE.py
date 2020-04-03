
"""
Created on Sun Mar 22 17:31:18 2020

@author: David Billingsley
"""

#This script applies the Guass-Newton algorithm to find the best fit parameters
#for a logistic curve of the form F(t) = A + (K - A)/(1 + e^(-B(t - M)))
#Guass-Newton algorithm here: https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm

import numpy as n
from numpy import linalg as LA

#these functions are partial derivatives of 
#the model F = A + (K - A)/(1 + e^(-B(t - M)))

def f(t, a, k, b, m):
    
    return a + (k - a)/(1 + n.exp(-b*(t - m)))

def dA(t, a, k, b, m):
    
    return 1/(n.exp(b*(t-m))+1)

def dK(t, a, k, b, m):
    
    return 1/(n.exp(b*(m-t))+1)

def dB(t, a, k, b, m):
    
    return -(k-a)*(m-t)*n.exp(-b*(t-m))/((n.exp(-b*(t-m))+1)**2)

def dM(t, a, k, b, m):
    
    return (k-a)*(-b)*n.exp(-b*(t-m))/((n.exp(-b*(t-m))+1)**2)


#fake numbers just for testing purposes as I write script
#independent variable
#ind = [1, 2, 3, 4, 5, 6, 7]
#observed = [7.572, 5.816, 5.717, 3.132, 2.121, 2.229, 2.215]
#A = 1.5
#B = -0.49
#K = 7.5
#M = 4.272
#params = [A, K, B, M]   

#gradient of F
grad = [dA, dK, dB, dM]

#parameter vector, starting at best guess

# Calculates Jacobian for a given set of parameters at points x of ind. var.
# x is your independent variables.
# params is your initial best guess for A, K, B, M 
# enter these at the command line
def jacobianMatrix(x, params):
    
    #Jacobian definition I am using defines each row as grad f at point x
    
    jf = n.zeros((len(x), 4))
    
    for i in range(len(x)):

         for j in range(4):
             
             #grad[i] is calling the ith partial derivative at the point 
             #(x[j], A, K, B, M)
             jf[i][j]= grad[j](x[i], params[0], params[1], params[2], params[3])
     
    print(jf.shape)
    
    return jf

#obs is your actual data at the independent variable x.
#x and obs should have the same length
def residuals(obs, x, params):
    
    r = []
    
    #calculate residual as difference between observed and model at each x
    for i in range(len(x)):
        
        r.append(obs[i] - f(x[i], params[0], params[1], params[2], params[3]))
    
    return r

#if params_i is the the ith iteration of params
#stop is the threshold at which |params_i - params_i+1|<stop.
#ie, when further iterations are not significantly improving on the previous ones
def gaussNewton(obs, x, params, stop):

    while True:
        
        old = params
        
        jf = jacobianMatrix(x, params)
        
        jfInv = LA.pinv(jf)
        
        params = n.add(old, n.matmul(jfInv, residuals(obs, x, old)))
        
        print(LA.norm(n.subtract(params, old)))
        print(params)
        
        if LA.norm(n.subtract(params, old)) <= stop:
            break

    return params


        
        
    
    
    
    
    
    


