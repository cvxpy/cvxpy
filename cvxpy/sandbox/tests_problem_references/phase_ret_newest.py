
import numpy as np

import cvxpy as cp

np.random.seed(0)
m, n = 100, 100
x_true = np.random.randn(n)
A = np.random.randn(m, n)
y = np.abs(A @ x_true)

# solve
x = cp.Variable((n, ))
obj = cp.sum(cp.square(cp.abs(A @ x) - y))
problem = cp.Problem(cp.Minimize(obj))

# very interesting that it crashes with zero initialization but not with 1e-6 initialization!
x.value = 1e-6 * np.ones(n)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
obj_value1 = np.sum((np.abs(A @ x.value) - y) ** 2)
recovered_x1 = x.value 

#print("true signs:      ", np.sign(A @ x_true))
#print("recovered signs: ", np.sign(A @ x.value))

# polishing
signs = np.sign(A @ x.value)
signs[np.abs(A @ x.value) <= 1e-5] = 0
t = cp.Variable((m, ), nonneg=True)
#new_obj = cp.sum(cp.square(signs * t - y))
new_obj = cp.sum_squares(t - y)
constraints = [cp.multiply(signs, t) == A @ x]
problem = cp.Problem(cp.Minimize(new_obj), constraints)
x.value = None
#x.value = np.random.rand(n) + 0.5
#problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
problem.solve(solver=cp.MOSEK, verbose=False)
obj_value2 = np.sum((np.abs(A @ x.value) - y) ** 2)
obj_value3 = new_obj.value
recovered_x2 = x.value


residual = np.linalg.norm(np.abs(A @ recovered_x1) - y)
print("residual: ", residual)

#print("true x:       ", np.round(x_true, 2))
#print("recovered_x1:  ", np.round(recovered_x1, 2))
#print("recovered_x2:  ", np.round(recovered_x2, 2))
#print("obj_value1: ", obj_value1)
#print("obj_value2: ", obj_value2)
#print("obj_value3: ", obj_value3)
#pdb.set_trace()