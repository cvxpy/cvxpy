import cvxpy as cp

print('Test div.')
x = cp.Variable()
obj = cp.Minimize(x / 5)
problem = cp.Problem(obj, [x == 5])
problem.solve(cp.SCS)
print(x.value)
print(problem.value)
