import cvxpy as cp

quad_over_lin = lambda x, t: cp.perspective(cp.sum_squares, x, t)

x = cp.Variable(10)
t = cp.Variable(1,)

objective = quad_over_lin(x, t)
prob = cp.Problem(cp.Minimize(objective))
prob.solve()