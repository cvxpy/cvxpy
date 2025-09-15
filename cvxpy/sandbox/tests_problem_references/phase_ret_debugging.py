
import pdb

import numpy as np
import numpy.linalg as LA

import cvxpy as cp

np.random.seed(2)
m, n = 100, 30
x_true = np.random.rand(n)
A = np.random.rand(m, n)
y = np.abs(A @ x_true) 
true_signs = np.sign(A @ x_true)


# solve
x = cp.Variable((n,))
t = cp.Variable((m,), nonneg=True)
obj = cp.sum(cp.square(t - y))
constraints = [cp.multiply(true_signs, t) == A @ x]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.MOSEK, verbose=True)
recovered_x1 = x.value 


residual = LA.norm(np.abs(A @ recovered_x1) - y)
print("residual: ", residual)

print("true x:       ", np.round(x_true, 2))
print("recovered_x1:  ", np.round(recovered_x1, 2))
pdb.set_trace()