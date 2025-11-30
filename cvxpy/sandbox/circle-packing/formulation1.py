


import numpy as np

import cvxpy as cp

rng = np.random.default_rng(5)

# number of circles
n = 10
radius = rng.uniform(1.0, 3.0, n)

# build problem
centers = cp.Variable((2, n), name='c')
constraints = []
for i in range(n - 1):
    for j in range(i + 1, n):
        constraints += [cp.sum(cp.square(centers[:, i] - centers[:, j])) >=
                         (radius[i] + radius[j]) ** 2]

# initialize centers to random locations
centers.value = rng.uniform(-5.0, 5.0, (2, n))

t = cp.Variable()

# formulation one
obj = cp.Minimize(t)
constraints += [cp.max(cp.norm_inf(centers, axis=0) + radius) <= t]
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.IPOPT, nlp=True, verbose=True, derivative_test='none',
              least_square_init_duals='no')

# compute manual residuals 
for i in range(n - 1):
    for j in range(i + 1, n):
        dist_ij = np.linalg.norm(centers.value[:, i] - centers.value[:, j])
        residual_ij = -(dist_ij ** 2 - (radius[i] + radius[j]) ** 2)
        print("residual between circles", i, j, ":", residual_ij)
        
print("centers formulation 1: \n", centers.value)