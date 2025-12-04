import numpy as np

import cvxpy as cp

np.random.seed(0)

# Problem dimensions
m, n = 50, 10
x_true = 10 * np.random.randn(n)
A = np.random.randn(m, n)
y = np.abs(A @ x_true) #+  np.random.randn(m)

x = cp.Variable(n)
# if I initialize from 0 IPOPT terminates after one iteration...
# this is extremely weird! must look into this
x0 =  2 * x_true
x.value = x0 
#obj = cp.sum_squares(cp.square(A @ x) - y ** 2)
#obj = cp.sum_squares(A @ x - y ** 2)
#pdb.set_trace()
#obj = cp.sum_squares(cp.abs(A @ x) - y)
t = cp.Variable(m)
v = cp.Variable(m)
#constraints = [t == cp.abs(v), v == A @ x]
constraints = [cp.square(t) == cp.square(v), v == A @ x, t >= 0]
obj = cp.sum_squares(t - y) 
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

print("x0: ", x0)
print("true x: ", x_true)
print("recovered x: ", x.value)
