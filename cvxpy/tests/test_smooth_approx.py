import cvxpy as cp

x = cp.Variable()
prob = cp.Problem(cp.Minimize(cp.abs(x)), [x >= 1])
prob.solve(solver=cp.IPOPT, nlp=True)
assert prob.status == cp.OPTIMAL
