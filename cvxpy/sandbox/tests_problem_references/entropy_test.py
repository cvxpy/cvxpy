import pdb

import numpy as np

import cvxpy as cp

np.random.seed(0)

n = 10
q = cp.Variable((n, ), nonneg=True)
A = np.random.rand(n, n)
q.value = np.random.rand(n)
q.value = q.value / np.sum(q.value)
obj = cp.sum(cp.entr(q))
constraints = [cp.sum(q) == 1]
problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
q_opt_nlp = q.value 

assert(np.sum(q_opt_nlp > 1e-8) == 1)

pdb.set_trace()

problem.solve(solver=cp.CLARABEL, verbose=True)
q_opt_clarabel = q.value

pdb.set_trace()

print("q_opt_nlp: ", q_opt_nlp)
print("q_opt_clarabel: ", q_opt_clarabel)

