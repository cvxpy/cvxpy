import pdb

import numpy as np

import cvxpy as cp

np.random.seed(0)

n = 50
p = np.random.rand(n, )
p = p / np.sum(p)
q = cp.Variable(n, nonneg=True)
A = np.random.rand(n, n)
obj = cp.sum(cp.rel_entr(A @ q, p))
constraints = [cp.sum(q) == 1]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
q_opt_nlp = q.value 

problem.solve(solver=cp.CLARABEL, verbose=True)
q_opt_clarabel = q.value

print("q_opt_nlp: ", q_opt_nlp)
print("q_opt_clarabel: ", q_opt_clarabel)

pdb.set_trace()

