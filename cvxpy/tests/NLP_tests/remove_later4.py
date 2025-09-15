import numpy as np
import cvxpy as cp

np.random.seed(0)
m, n = 5, 2
b = cp.Constant(np.ones(m), name="b")
rand = np.random.randn(m - 2*n, n)
A = cp.Constant(np.vstack((rand, np.eye(n), np.eye(n) * -1)), name="A")
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
assert problem.status == cp.OPTIMAL
