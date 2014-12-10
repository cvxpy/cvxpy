from cvxpy import *
import numpy as np
m = 200
n = 100
k = 100
np.random.seed(2)
A = np.random.randn(m, n)
# A = np.ones((m, n))
X = Variable(n, k)

cost = norm(X, 'fro')
prob = Problem(Minimize(cost), [A*X >= 0])
print prob.solve(solver=ECOS, verbose=True)
print prob.solve(solver=SCS_MAT_FREE, verbose=True,
                 max_iters=2500)
