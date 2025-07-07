import cvxpy as cp

# Define variables
x = cp.Variable(1)
y = cp.Variable(1)

objective = cp.Minimize(cp.maximum(x, y))

constraints = [x - 14 == 0, y - 6 == 0]

problem = cp.Problem(objective, constraints)
print(cp.installed_solvers())
problem.solve(solver=cp.CLARABEL, verbose=True)
#problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
print(x.value, y.value)
print(problem.status)
print(problem.value)
