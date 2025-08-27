import numpy as np

import cvxpy as cp

np.random.seed(1)

n, m, k = 5, 4, 2
noise_level = 0.05
X_true = np.random.rand(n, k)
Y_true = np.random.rand(k, m)
A_noisy = X_true @ Y_true + noise_level * np.random.randn(n, m)
A_noise = np.clip(A_noisy, 0, None)
# initialize X and Y to random nonnegative values
X = cp.Variable((n, k), bounds=[0, np.inf])
X.value = np.ones((n, k))
Y = cp.Variable((k, m), bounds=[0, np.inf])
Y.value = np.ones((k, m))
obj = cp.sum(cp.square(A_noise - X @ Y))
prob = cp.Problem(cp.Minimize(obj))
prob.solve(solver=cp.IPOPT, nlp=True, verbose=True)
