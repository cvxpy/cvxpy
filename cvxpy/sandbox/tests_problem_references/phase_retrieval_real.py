import numpy as np

import cvxpy as cp

np.random.seed(0)

m, n = 10, 5
x_true = np.random.randn(n)
A = np.random.randn(m, n)
y = np.abs(A @ x_true) 
x = cp.Variable(n)


obj = cp.sum_squares(cp.square(A @ x) - y ** 2)
problem = cp.Problem(cp.Minimize(obj))

# With this initilaization IPOPT terminates after one iteration
x.value = 0 * np.ones(n)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_star_one = x.value

# With this initialization IPOPT converges to the x_true
x.value = 1e-8 * np.ones(n)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_star_two = x.value

print("true x:                                 ", np.round(x_true, 2))
print("recovered x with zero initialization:   ", np.round(x_star_one, 2))
print("recovered x with small initialization:  ", np.round(x_star_two, 2))
