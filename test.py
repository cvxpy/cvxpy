import cvxpy as cp
import numpy as np
np.random.seed(1)
m = 4
n = 2
k = 1
A = np.random.randn(m, n)
xstar = np.random.randn(n, k)
b = A@xstar
alpha = cp.Parameter()
x = cp.Variable((n, k))
alpha.value = 1
prob = cp.Problem(cp.Minimize(alpha + cp.sum(cp.norm(A*x - b, 2, axis=0))))
result = prob.solve(solver=cp.SCS, verbose=True)
print(result)
print(x.value)
