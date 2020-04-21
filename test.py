import cvxpy as c

x = [[4144.30127531]]
y = [[7202.52114311]]
z = c.Variable(shape=(1, 1))
objective = c.Minimize(c.quad_form(z, x) - 2 * z.T @ y) 

prob = c.Problem(objective)
prob.solve('OSQP')
