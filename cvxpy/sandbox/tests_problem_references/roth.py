import numpy as np

import cvxpy as cp

x = cp.Variable(2, name='x')
F = cp.hstack([-13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1],
               -29 + x[0] + ((x[1] + 1) * x[1] - 14) * x[1]])

# formulation 1 
objective = cp.sum_squares(F)
problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([0.5, -2])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_opt_one = x.value

# formulation 2
objective = cp.square(-13 + x[0] + ((5 - x[1]) * x[1] - 2) * x[1]) + \
            cp.square(-29 + x[0] + ((x[1] + 1) * x[1] - 14) * x[1])
problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([0.5, -2])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_opt_two = x.value

print(x_opt_one)
print(x_opt_two)