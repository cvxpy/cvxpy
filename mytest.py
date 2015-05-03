import cvxpy as cvx
x = cvx.Variable(3, name='x')
y = cvx.Variable(1, name='y')
p = 0
g = cvx.power(x, p)


prob = cvx.Problem(cvx.Maximize(cvx.sum_entries(g) + y), [0 <= x, x <= 1, y <= 2])
prob.solve()