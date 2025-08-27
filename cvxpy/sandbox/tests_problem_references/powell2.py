import pdb

import numpy as np

import cvxpy as cp

x = cp.Variable(2, name='x')
F = cp.hstack([1e4*x[0]*x[1] - 1,
               cp.exp(-x[0]) + cp.exp(-x[1]) - 1.001])

# formulation 1 
objective = cp.sum_squares(F)
problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([0, 1])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# formulation 2 ()
objective = cp.square(1e4*x[0]*x[1] - 1) + cp.square(cp.exp(-x[0]) + cp.exp(-x[1]) - 1.001)
problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([0, 1])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

pdb.set_trace()