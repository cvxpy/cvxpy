import cvxpy as cp
import numpy as np

k = 5
x = cp.Variable((k, k))
x.value = np.ones((k, k)) / k**2
left = np.ones(k) / k
right = np.exp(-np.linspace(-1, 1, k)**2)
right = right / np.sum(right)
obj = cp.entr(x).sum()
constr = [cp.sum(x, axis=1) == right,
            cp.sum(x, axis=0) == left]
problem = cp.Problem(cp.Maximize(obj), constr)
problem.solve(solver=cp.IPOPT, nlp=True)
assert problem.status == cp.OPTIMAL
