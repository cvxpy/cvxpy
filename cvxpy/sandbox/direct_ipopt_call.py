import cvxpy as cp

# Define variables
x = cp.Variable(3)
y = cp.Variable(3)

objective = cp.Maximize(cp.sum(cp.maximum(x, y)))

constraints = [x <= 14, y <= 6]

problem = cp.Problem(objective, constraints)
print(cp.installed_solvers())
#problem.solve(solver=cp.CLARABEL, verbose=True)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
print(x.value, y.value)
print(problem.status)
print(problem.value)
