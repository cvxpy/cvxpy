import numpy as np

import cvxpy as cp

x = cp.Variable(4, name='x')
F = cp.hstack([x[0] + 10 * x[1],
               np.sqrt(5) * (x[2] - x[3]),
               cp.square(x[1] - 2 * x[2]),
               np.sqrt(10) * (x[0] - x[3])])


# formulation 1 - works (95 iterations)
objective = cp.sum_squares(F)
problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([3, -1, 0, 1])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# formulation 2 - works (41 iterations)
objective = cp.square(x[0] + 10 * x[1]) + \
            cp.square(np.sqrt(5) * (x[2] - x[3])) + \
            cp.power(x[1] - 2 * x[2], 4) + \
            cp.square(np.sqrt(10) * (x[0] - x[3]))

problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([3, -1, 0, 1])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)


# formulation 3 - works (18 iterations)
objective = cp.square(x[0] + 10 * x[1]) + \
            cp.square(np.sqrt(5) * (x[2] - x[3])) + \
            cp.square(cp.square(x[1] - 2 * x[2])) + \
            cp.square(np.sqrt(10) * (x[0] - x[3]))

problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([3, -1, 0, 1])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# formulation 4 - works (18 iterations)
objective = cp.power(x[0] + 10 * x[1], 2) + \
            cp.power(np.sqrt(5) * (x[2] - x[3]), 2) + \
            cp.square(cp.square(x[1] - 2 * x[2])) + \
            cp.power(np.sqrt(10) * (x[0] - x[3]), 2)

problem = cp.Problem(cp.Minimize(objective))
x.value = np.array([3, -1, 0, 1])
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# formulation 5 - canonicalization fails
#objective = cp.norm(F)
#problem = cp.Problem(cp.Minimize(objective))
#problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
#pdb.set_trace()