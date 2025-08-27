
import numpy as np

import cvxpy as cp

np.random.seed(0)

m, n = 300, 50
x_true = np.random.randn(n)
A = np.random.randn(m, n)
y = np.abs(A @ x_true) 
x = cp.Variable(n)

t = cp.Variable(m)
v = cp.Variable(m)

#obj = cp.norm1(cp.square(A @ x) - y ** 2)
obj = cp.sum(t)
constraints = [cp.square(t) == cp.square(v), v == cp.square(A @ x) - y ** 2, t >= 0]
problem = cp.Problem(cp.Minimize(obj), constraints)

# With this initialization IPOPT terminates after one iteration
x.value = 0 * np.ones(n)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_star_one = x.value

# With this initialization IPOPT converges to the x_true
x.value = np.ones(n)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_star_two = x.value

print("true x:                                 ", np.round(x_true, 2))
print("recovered x with zero initialization:   ", np.round(x_star_one, 2))
print("recovered x with small initialization:  ", np.round(x_star_two, 2))
