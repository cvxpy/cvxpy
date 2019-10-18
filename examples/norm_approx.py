"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import sys

import cvxopt
import numpy as np
from pylab import *
import math

from cvxpy import *

# Taken from CVX website http://cvxr.com/cvx/examples/
# Examples 5.6,5.8: An l_p norm approximation problem
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Boyd & Vandenberghe "Convex Optimization"
# Joelle Skaf - 08/23/05
#
# The goal is to show the following problem formulations give all the same
# optimal residual norm ||Ax - b||:
# 1)        minimize    ||Ax - b||
# 2)        minimize    ||y||
#               s.t.    Ax - b = y
# 3)        maximize    b'v
#               s.t.    ||v||* <= 1  , A'v = 0
# 4)        minimize    1/2 ||y||^2
#               s.t.    Ax - b = y
# 5)        maximize    -1/2||v||*^2 + b'v
#               s.t.    A'v = 0
# where ||.||* denotes the dual norm of ||.||

# Input data
n = 4
m = 2*n
A = cvxopt.normal(m,n)
b = cvxopt.normal(m,1)
p = 2
q = p/(p-1)


# Original problem
x = Variable(n)
objective1 = Minimize( norm ( A*x - b , p) )
p1 = Problem(objective1, [])
print('Computing the optimal solution of problem 1... ')
opt1 = p1.solve()

# Reformulation 1
x = Variable(n)
y = Variable(m)
objective2 = Minimize ( norm( y, p ) )
p2 = Problem(objective2, [ A*x - b == y ])
print('Computing the optimal solution of problem 2... ')
opt2 = p2.solve()

# Dual of reformulation 1
nu = Variable(m)
objective3 = Maximize( b.T * nu )
p3 = Problem(objective3, [ norm( nu, q) <= 1, A.T*nu == 0 ])
print('Computing the optimal solution of problem 3... ')
opt3 = p3.solve()

# Reformulation 2
x = Variable(n)
y = Variable(m)
objective4 = Minimize( 0.5*square( norm(y, p) ) )
p4 = Problem(objective4, [ A*x - b == y ] )
print('Computing the optimal solution of problem 4... ')
opt4 = math.sqrt(2*p4.solve())

# Dual of reformulation 2
nu = Variable(m)
objective5 = Maximize( -0.5*square( norm(nu,q) ) + b.T*nu )
p5 = Problem(objective5, [ A.T*nu==0 ])
print('Computing the optimal solution of problem 5... ')
opt5 = math.sqrt(2*p5.solve())

# Display results
print('------------------------------------------------------------------------')
print('The optimal residual values for problems 1,2,3,4 and 5 are respectively:')
print(opt1, opt2, opt3, opt4, opt5)
print('They are equal as expected!')
