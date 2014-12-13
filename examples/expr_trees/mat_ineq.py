from cvxpy import *
import numpy as np
m = 200
n = 200
k = 1000
np.random.seed(2)
A = np.random.randn(m, n)
# A = np.ones((m, n))
X = Variable(n, k)

cost = norm(X, 'fro')
prob = Problem(Minimize(cost), [A*X >= 1])
print prob.solve(solver=ECOS, verbose=True)
print prob.solve(solver=SCS_MAT_FREE, verbose=True,
                 max_iters=1000, equil_steps=1, cg_rate=2)

# Results:
# m=n=k=200, ECOS 37.94 sec, SCS_MAT_FREE 2.42e+01s
