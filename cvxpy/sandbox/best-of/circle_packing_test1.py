import numpy as np

import cvxpy as cp

# data generation
rng = np.random.default_rng(5)
n = 10
radius = rng.uniform(1.0, 3.0, n)
init_centers = rng.uniform(-5.0, 5.0, (n, 2))

# build optimization problem
centers = cp.Variable((n, 2), name='c')
constraints = []
for i in range(n - 1):
    constraints += [cp.sum((centers[i, :] - centers[i+1:, :]) ** 2, axis=1) >=
                     (radius[i] + radius[i+1:]) ** 2]
obj = cp.Minimize(cp.max(cp.norm_inf(centers, axis=1) + radius))
prob = cp.Problem(obj, constraints)

# solve
centers.value = init_centers
centers.sample_bounds = [-5.0, 5.0]  
prob.solve(nlp=True, verbose=True, derivative_test='none', best_of=10)
