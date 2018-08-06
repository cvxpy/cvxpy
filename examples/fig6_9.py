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
from scipy import sparse
from pylab import *
import math

from cvxpy import *
from multiprocessing import Pool

# Taken from CVX website http://cvxr.com/cvx/examples/
# Figure 6.9: An optimal tradeoff curve
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Section 6.3.3
# Boyd & Vandenberghe "Convex Optimization"
# Original by Lieven Vandenberghe
# Adapted for CVX Joelle Skaf - 09/29/05
# (a figure is generated)
#
# Plots the optimal trade-off curve between ||Dx||_2 and ||x-x_cor||_2 by
# solving the following problem for different values of delta:
#           minimize    ||x - x_cor||^2 + delta*||Dx||^2
# where x_cor is the a problem parameter, ||Dx|| is a measure of smoothness

# Input data
n = 400
t = np.array(range(0,n))

exact = 0.5*sin(2*np.pi*t/n) * sin(0.01*t)
corrupt = exact + 0.05 * np.random.randn(len(exact))
corrupt = cvxopt.matrix(corrupt)

e = np.ones(n).T
ee = np.column_stack((-e,e)).T
D = sparse.spdiags(ee, range(-1,1), n, n)
D = D.todense()
D = cvxopt.matrix(D)

# Solve in parallel
nopts = 10
lambdas = np.linspace(0, 50, nopts)
# Frame the problem with a parameter
lamb = Parameter(nonneg=True)
x = Variable(n)
p = Problem( Minimize( norm(x-corrupt) + norm(D*x) * lamb ) )


# For a value of lambda g, we solve the problem
# Returns [ ||Dx||_2 and ||x-x_cor||_2 ]
def get_value(g):
	lamb.value = g
	result = p.solve()
	return [np.linalg.norm( x.value - corrupt ), np.linalg.norm(D*x.value) ]


pool = Pool(processes = 4)
# compute allocation in parallel
norms1, norms2 = zip(*pool.map(get_value, lambdas))

plot(norms1, norms2)
xlabel('||x - x_{cor}||_2')
ylabel('||Dx||_2')
title('Optimal trade-off curve')
show()
