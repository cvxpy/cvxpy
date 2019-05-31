import cvxpy as cp

# DGP requires Variables to be declared positive via `pos=True`.
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
z = cp.Variable(pos=True)

objective_fn = x * y * z
constraints = [
  4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
problem = cp.Problem(cp.Maximize(objective_fn), constraints)
cp.solvers.defines.INSTALLED_SOLVERS = ['ECOS', 'ECOS_BB', 'CVXOPT', 'GLPK', 'GLPK_MI', 'SCS', 'GUROBI', 'OSQP']
cp.solvers.defines.INSTALLED_CONIC_SOLVERS += ['GUROBI']
#cp.solvers.defines.QP_SOLVERS += ['GUROBI']
problem.solve(gp=True, verbose=True)
print("Optimal value: ", problem.value)
print("x: ", x.value)
print("y: ", y.value)
print("z: ", z.value)
