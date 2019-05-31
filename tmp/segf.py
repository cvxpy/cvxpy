import cvxpy as cvx
x = cvx.Variable((1, 1))
expr0 = x * 1  # segfaults iff indexed
expr1 = x * cvx.Constant(1) # segfaults iff indexed
expr2 = x * 1 + 0 # works regardless of indexing

expr = expr0  # adjust this value during testing (try expr0, expr1, expr2, or more if you'd like)

prob = cvx.Problem(cvx.Minimize(expr), [])
prob.solve()
print('first is ok')
other_expr = expr[0,0]
prob = cvx.Problem(cvx.Minimize(other_expr), [])
prob.solve()
print('second is ok')
