
import pdb

import numpy as np

import cvxpy as cp

np.random.seed(1)
m, n = 10, 3
x_true = np.random.randn(n)
A = np.random.randn(m, n)
y = np.abs(A @ x_true) 

# solve
x = cp.Variable((n, ), name='x')
obj = cp.sum(cp.abs(cp.square(A @ x) - y ** 2))
problem = cp.Problem(cp.Minimize(obj))
x.value = np.ones(n)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
obj_value1 = np.sum(np.abs(np.square(A @ x.value) - y ** 2))
recovered1 = x.value

# polishing step
signs = np.sign((cp.square(A @ x) - y ** 2).value)
t = cp.Variable((m, ), nonneg=True)
new_obj = cp.sum(t)
constraints = [cp.multiply(signs, t) == cp.square(A @ x) - y ** 2]
problem = cp.Problem(cp.Minimize(new_obj), constraints)
#x.value = np.random.rand(n) + 0.5
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
obj_value2 = np.sum(np.abs(np.square(A @ x.value) - y ** 2))
recovered2 = x.value


print("obj_value1: ", obj_value1)
print("obj_value2: ", obj_value2)


# With this initialization IPOPT converges to the x_true
#x.value = np.ones(n)
#x.value = np.random.rand(n) + 0.5

#x0 = np.linalg.lstsq(A, y, rcond=None)[0]
#x.value = x0
#x.value = x_true


print("true x:       ", np.round(x_true, 2))
print("recovered x:  ", np.round(recovered1, 2))
print("recovered x2: ", np.round(recovered2, 2))
pdb.set_trace()