import pdb

import numpy as np

import cvxpy as cp

np.random.seed(0)

c_true = 0.1
a_true = 0.2 
x = np.arange(0, 10, 1)
y = c_true * np.exp(a_true * x) #+ 0.1 * np.random.randn(*x.shape)
c = cp.Variable()
a = cp.Variable()
obj = cp.sum(cp.square(y - c * cp.exp(a * x)))
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

pdb.set_trace()