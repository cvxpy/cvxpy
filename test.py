import cvxpy as cp

x = cp.Variable()
obj = cp.Minimize(cp.square(x))
constraints = [x == x]
problem = cp.Problem(obj, constraints)
problem.solve(cp.ECOS)
