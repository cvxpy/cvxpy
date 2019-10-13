import cvxpy as cp

x = cp.Variable()
y = cp.Variable()
obj = cp.Minimize(0)
constr = [x - x == 0]
problem = cp.Problem(obj, constr)
problem.solve(solver=cp.ECOS)
print('done')
