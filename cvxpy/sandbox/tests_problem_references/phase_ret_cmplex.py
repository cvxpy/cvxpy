
import numpy as np

import cvxpy as cp

np.random.seed(0)

# Problem dimensions
m, n = 50, 10
x_true = 10 * (np.random.randn(n) + 1j*np.random.randn(n))
A = np.random.randn(m, n) + 1j*np.random.randn(m, n)
y = np.abs(A @ x_true)

x = cp.Variable(n, complex=True)
x0 = np.random.rand(n) + 1j*np.random.rand(n)
x.value = x0 
obj = cp.sum_squares(cp.abs(A @ x) - y)
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

print("x0: ", x0)
print("true x: ", x_true)
print("recovered x: ", x.value)
