import cvxpy as cp
import numpy as np
np.random.seed(1)
m = 10
n = 5
A = np.abs(np.random.randn(m, n))
X = cp.Variable((2, m))

expr = X@A
# expr = (A.T@X.T).T
prob = cp.Problem(cp.Minimize(cp.sum(expr)), [X >= 2])
result = prob.solve(solver=cp.SCS)
print(result)

expr = (A.T@X.T).T
prob = cp.Problem(cp.Minimize(cp.sum(expr)), [X >= 2])
result = prob.solve(solver=cp.SCS)
print(result)

C = cp.Variable((3, 3))
obj = cp.Maximize(C[0, 2])
constraints = [cp.diag(C) == 1,
                C[0, 1] == 0.6,
                C[1, 2] == -0.3,
                C == C.T,
                C >> 0]
prob = cp.Problem(obj, constraints)
result = prob.solve(solver=cp.SCS, verbose=True)
print(result, 0.583151)
