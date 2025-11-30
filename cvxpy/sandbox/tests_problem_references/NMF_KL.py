import numpy as np

import cvxpy as cp

np.random.seed(0)


n, m, k = 40, 20, 4
X_true = np.random.rand(n, k)
Y_true = np.random.rand(k, m)
A = X_true @ Y_true 
A = np.clip(A, 0, None)
X = cp.Variable((n, k), bounds=[0, None], name='X')
Y = cp.Variable((k, m), bounds=[0, None], name='Y')
X.value = np.random.rand(n, k)
Y.value = np.random.rand(k, m)
obj = cp.sum(cp.kl_div(A, X @ Y))
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
assert(obj.value <= 1e-10)
