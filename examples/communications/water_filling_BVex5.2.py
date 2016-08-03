#!/usr/bin/env python3
# @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
# Boyd and Vandenberghe, Convex Optimization, example 5.2 page 145

import numpy as np
import cvxpy as cvx

'''
Input parameters   
  n: Number of communication channels or 'buckets'
  a: Floor above the baseline for each channel at which power can be added
  sum_x: Total power to be allocated to the n channels
'''

def water_filling(n,a,sum_x=1):
  '''
Boyd and Vandenberghe, Convex Optimization, example 5.2 page 145
Water-filling.
  
This problem arises in information theory, in allocating power to a set of
n communication channels in order to maximise the total channel capacity.
The variable x_i represents the transmitter power allocated to the ith channel, 
and log(α_i+x_i) gives the capacity or maximum communication rate of the channel. 
The objective is to minimize  -∑log(α_i+x_i) subject to the constraint ∑x_i = 1 
  '''
  # Declare variables and parameters
  x = cvx.Variable(n)
  alpha = cvx.Parameter(n,sign='positive')
  alpha.value = a
  #alpha.value = np.ones(n)
  # Choose objective function. Interpret as maximising the total communication rate of all the channels
  obj = cvx.Maximize(cvx.sum_entries(cvx.log(alpha + x)))
  # Declare constraints
  constraints = [x >= 0, cvx.sum_entries(x) - sum_x == 0]
  # Solve
  prob = cvx.Problem(obj, constraints)
  prob.solve()
  if(prob.status=='optimal'):
    return prob.status,prob.value,x.value
  else:
    return prob.status,np.nan,np.nan

# as an example, solve the water filling problem for three buckets
if __name__ == '__main__':
  print(water_filling.__doc__)
  # print all arrays to 3 significant figures
  np.set_printoptions(precision=3)
  buckets=3
  alpha = np.array([0.8,1.0,1.2])
  print('Number of buckets = %s'%buckets)
  stat,prob,x=water_filling(buckets,alpha)
  print('Problem status: ',stat)
  print('Optimal communication rate = %.4g '%prob)
  print('Transmitter powers:\n', x)
