import cvxpy as cp
import numpy as np

A = np.random.randn(20, 20)
b = A @ np.random.randn(20)

x = cp.Variable(20)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
prob.solve(solver=cp.OSQP, max_iter=1)
