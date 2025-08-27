import numpy as np

import cvxpy as cp

m = 100
n = 5
Ps = []
Vs = []
x_true = np.random.randn(n)
for i in range(m):
    P = np.random.randn(n, n)
    P = P + P.T
    Ps.append(P)
    v = x_true.T @ P @ x_true
    Vs.append(v)

x = cp.Variable(n)
obj = 0
for i in range(m):
    obj += cp.square(cp.quad_form(x, Ps[i]) - Vs[i])

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
