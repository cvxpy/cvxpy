import cvxpy as cvx
import numpy as np


x = cvx.Variable(5)
y = cvx.Variable()
f = cvx.quad_over_lin(x, y)

x = cvx.Variable((3, 5))
y = cvx.Variable(3)
f = cvx.quad_over_lin(x, y)

#x = cvx.Variable(3)
#y = cvx.Variable(3)
#f =cvx.quad_over_lin(x, y)

constr = []
constr += [np.array([1, 2, 3]) == y]
#constr += [y == 1]
constr += [x == 1]
prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.quad_over_lin(x, y))), constr)
prob.solve()
print(prob.value)
