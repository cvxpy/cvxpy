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
