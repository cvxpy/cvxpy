import numpy as np

import cvxpy as cp

# x = cp.Variable()
x = cp.Variable(5)
# TODO: support multiple powers
# prob = cp.Problem(cp.Minimize(cp.power(x,4)), [x >= 1])
constr = [cp.power(x, 4.2, _approx=True) <= np.ones(5)]
prob = cp.Problem(cp.Minimize(x[0]+x[1]-x[2]), constr)
prob.solve()
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal var:", x.value)