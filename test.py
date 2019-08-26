import numpy as np
import cvxpy

x = cvxpy.Variable(shape=(5, 10))
cons = [cvxpy.pnorm(x, 'inf', axis=0) <= 1]
c = np.random.randn(5, 10)
objective = cvxpy.Minimize(cvxpy.trace(c.T @ x))
prob = cvxpy.Problem(objective, cons)

prob.solve(solver='ECOS', verbose=True)
