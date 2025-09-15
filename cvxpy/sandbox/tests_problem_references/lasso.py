
import numpy as np

import cvxpy as cp

np.random.seed(1)
m, n = 30, 100
b = np.random.randn(m)
A = np.random.randn(m, n)

lmbda = 1

x = cp.Variable((n, ), name='x')
obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.CLARABEL, verbose=True)
x_star_dcp = x.value
obj_star_dcp = obj.value


x = cp.Variable((n, ), name='x')
obj = cp.sum(cp.square((A @ x - b))) + lmbda * cp.sum(cp.abs(x))
problem = cp.Problem(cp.Minimize(obj))
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
x_star_nlp = x.value
obj_star_nlp = obj.value

print("obj_star_nlp: ", obj_star_nlp)
print("obj_star_dcp: ", obj_star_dcp)
#pdb.set_trace()
