"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from cvxpy import *

# Based on formulation described
#    @ https://en.wikipedia.org/wiki/Travelling_salesman_problem (February 2016)

np.random.seed(1)

N = 5
distances = np.random.rand(N, N)
distances = (distances + distances.T)/2  # make symmetric = symmetric-TSP

# VARS
x = Variable((N, N), boolean=True)
u = Variable(N, integer=True)

# CONSTRAINTS
constraints = []
for j in range(N):
    indices = np.hstack((np.arange(0, j), np.arange(j + 1, N)))
    constraints.append(sum(x[indices, j]) == 1)
for i in range(N):
    indices = np.hstack((np.arange(0, i), np.arange(i + 1, N)))
    constraints.append(sum(x[i, indices]) == 1)

for i in range(1, N):
    for j in range(1, N):
        if i != j:
            constraints.append(u[i] - u[j] + N*x[i, j] <= N-1)

# OBJ
obj = Minimize(sum(multiply(distances, x)))

# SOLVE
prob = Problem(obj, constraints)
prob.solve(verbose=True)
print(prob.value)
