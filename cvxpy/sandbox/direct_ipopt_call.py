import cvxpy as cp

# Define variables
x = cp.Variable(1)
y = cp.Variable(1)
t = cp.Variable(1)

objective = cp.Minimize(-t)

constraints = [(t - x) * (t - y) == 0, t >= x, t >= y, x - 14 == 0, y - 6 == 0]

problem = cp.Problem(objective, constraints)
print(cp.installed_solvers())
problem.solve(solver=cp.IPOPT, nlp=True)
