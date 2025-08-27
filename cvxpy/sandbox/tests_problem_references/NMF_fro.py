import numpy as np

import cvxpy as cp

np.random.seed(0)

n, m, k = 5, 4, 2
noise_level = 0.05
X_true = np.random.rand(n, k)
Y_true = np.random.rand(k, m)
A_noisy = X_true @ Y_true + noise_level * np.random.randn(n, m)
A_noisy = np.clip(A_noisy, 0, None)
X = cp.Variable((n, k), nonneg=True)
Y = cp.Variable((k, m), nonneg=True)
obj = cp.sum_squares(A_noisy - X @ Y)
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)