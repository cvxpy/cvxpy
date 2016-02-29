import numpy as np
from cvxpy import *

# Based on formulation described
#    @ https://en.wikipedia.org/wiki/Travelling_salesman_problem (February 2016)

np.random.seed(1)

N = 10
distances = np.random.rand(N, N)
distances = (distances + distances.T)/2  # make symmetric = symmetric-TSP

# VARS
x = Bool(N, N)
u = Int(N)

# CONSTRAINTS
constraints = []
for j in range(N):
    indices = range(0, j) + range(j + 1, N)
    constraints.append(sum_entries(x[indices, j]) == 1)
for i in range(N):
    indices = range(0, i) + range(i + 1, N)
    constraints.append(sum_entries(x[i, indices]) == 1)

for i in range(1, N):
    for j in range(1, N):
        if i != j:
            constraints.append(u[i] - u[j] + N*x[i, j] <= N-1)

# OBJ
obj = Minimize(sum_entries(mul_elemwise(distances, x)))

# SOLVE
prob = Problem(obj, constraints)
prob.solve(verbose=True)
print prob.value
