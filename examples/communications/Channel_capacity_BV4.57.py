#!/usr/bin/env python3
# @author: R. Gowers, S. Al-Izzi, T. Pollington, R. Hill & K. Briggs
# Boyd and Vandenberghe, Convex Optimization, exercise 4.57 page 207

import cvxpy as cvx
import numpy as np

'''
Input parameters
  P: channel transition matrix P_ij(t) = P(output|input) at time t
  n: size of input
  m: size of output
'''

def channel_capacity(n,m,sum_x=1):
  '''
Boyd and Vandenberghe, Convex Optimization, exercise 4.57 page 207
Capacity of a communication channel.
  
We consider a communication channel, with input x(t)∈{1,..,n} and
output Y(t)∈{1,...,m}, for t=1,2,... .The relation between the
input and output is given statistically:
p_(i,j) = ℙ(Y(t)=i|X(t)=j), i=1,..,m  j=1,...,m
The matrix P ∈ ℝ^(m*n) is called the channel transition matrix, and
the channel is called a discrete memoryless channel. Assuming X has a
probability distribution denoted x ∈ ℝ^n, i.e.,
x_j = ℙ(X=j), j=1,...,n
The mutual information between X and Y is given by
∑(∑(x_j p_(i,j)log_2(p_(i,j)/∑(x_k p_(i,k)))))
Then channel capacity C is given by
C = sup I(X;Y).
With a variable change of y = Px this becomes
I(X;Y)=  c^T x - ∑(y_i log_2 y_i)
where c_j = ∑(p_(i,j)log_2(p_(i,j)))
  '''
  # n is the number of different input values
  # m is the number of different output values
  if n*m == 0:
    print('The range of both input and output values must be greater than zero')
    return 'failed',np.nan,np.nan
  # P is the channel transition matrix
  P = np.ones((m,n))
  # x is probability distribution of the input signal X(t)
  x = cvx.Variable(rows=n,cols=1)
  # y is the probability distribution of the output signal Y(t)
  y = P*x
  # I is the mutual information between x and y
  c = np.sum(P*np.log2(P),axis=0)
  I = c*x + cvx.sum_entries(cvx.entr(y))
  # Channel capacity maximised by maximising the mutual information
  obj = cvx.Minimize(-I)
  constraints = [cvx.sum_entries(x) == sum_x,x >= 0]
  # Form and solve problem
  prob = cvx.Problem(obj,constraints)
  prob.solve()
  if prob.status=='optimal':
    return prob.status,prob.value,x.value
  else:
    return prob.status,np.nan,np.nan

# as an example, let's optimise the channel capacity for two different possible input and output values
if __name__ == '__main__':
  print(channel_capacity.__doc__)
  # print all arrays to have 3 significant figures after the decimal place
  np.set_printoptions(precision=3)
  n = 2
  m = 2
  print('Number of input values=%s'%n)
  print('Number of outputs=%s'%m)
  stat,C,x=channel_capacity(n,m)
  print('Problem status ',stat)
  print('Optimal value of C = %.4g'%(C))
  print('Optimal variable x = \n', x)
