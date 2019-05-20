import numpy as np
import cvxpy as cp
m = 4
n = 2
k = 2
np.random.seed(1)
A = np.random.randn(m, n)
param_A = cp.Parameter((m, n), value=A)
xstar = np.random.randn(n)
b = A@xstar
print("A")
print(A)
print("b")
print(b)
alpha = cp.Parameter()
beta = cp.Parameter(nonneg=True)
x = cp.Variable(n)
t = cp.Variable()
prob = cp.Problem(cp.Minimize(alpha + alpha + beta*t),
                  [alpha*t >= cp.maximum(0, alpha),
                   alpha*t == alpha])

prob = cp.Problem(cp.Minimize(cp.norm2(param_A*x - b)),
                              [cp.diag(x) >> 0])
alpha.value = 1
beta.value = 2
result = prob.solve(solver=cp.SCS, verbose=True)
print(result)
print(x.value)
print(cp.norm2(A*x - b).value)
