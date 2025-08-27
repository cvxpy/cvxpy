import pdb

import numpy as np

import cvxpy as cp

n, m = 100, 25
A = np.random.randn(m, n)
b = np.random.randn(m)

x = cp.Variable(n)
obj = cp.Minimize(cp.sum_squares(A @ x - b))
constraints = [cp.square(x) == 4]
problem = cp.Problem(obj, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

pdb.set_trace()