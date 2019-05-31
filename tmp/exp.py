import cvxpy as cp


x = cp.Variable()
prob = cp.Problem(cp.Minimize(cp.exp(x)), [cp.ceil(x) <= 10])
prob.solve(qcp=True, verbose=True)
