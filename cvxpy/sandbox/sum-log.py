# write a log sum problem in cvxpy
import numpy as np
from scipy.io import savemat

import cvxpy as cp

# Generate random data
np.random.seed(0)
m, n = 500, 40
b = np.ones(m)
rand = np.random.randn(m - 2*n, n)
A = np.vstack((rand, np.eye(n), np.eye(n) * -1))
"""
m, n = 5, 2
A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [-0.5, 1]])
b = np.array([1, 1, 1, 1, 0.5])
"""
# Define the variable
x = cp.Variable(n)
# set initial value for x
objective = cp.Minimize(-cp.sum(cp.log(b - A @ x)))
problem = cp.Problem(objective, [])
# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True)
print("Optimal value:", problem.value)
print("Optimal x:", x.value)

problem.solve(solver=cp.CLARABEL)
print("Optimal value:", problem.value)
print("Optimal x:", x.value)

# Save the results for matlab
# savemat("analytic_center.mat", {"A": A, "b": b, "x": x.value})
np.savez("analytic_center.npz", A=A, b=b, x=x.value)
