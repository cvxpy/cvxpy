import numpy as np

import cvxpy as cp

np.random.seed(0)

n, m, k = 4, 4, 4
noise_level = 0.05
X_true = np.random.rand(n, k)
Y_true = np.random.rand(k, m)
A_noisy = X_true @ Y_true #+ noise_level * np.random.randn(n, m)
A_noisy = np.clip(A_noisy, 0, None)
X = cp.Variable((n, k), bounds=[1, None], name='X')
Y = cp.Variable((k, m), bounds=[1, None], name='Y')

# if we don't specify a value on this problem, it does not converge to the global minimizer
#X.value = np.random.rand(n, k)  
#Y.value = np.random.rand(k, m)  + 0.4
obj = cp.sum(cp.square(A_noisy - X @ Y))
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
