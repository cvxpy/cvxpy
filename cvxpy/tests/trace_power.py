import numpy as np

import cvxpy as cp

# x = cp.Variable()
x = cp.Variable(5)
# TODO: support multiple powers
# prob = cp.Problem(cp.Minimize(cp.power(x,4)), [x >= 1])
constr = [cp.power(x, 4.4) <= np.ones(5)]
prob = cp.Problem(cp.Minimize(x[0]+x[1]-x[2]), constr)
# prob = cp.Problem(cp.Minimize(x), [cp.power(x, 1.001, _approx=False) <= 1])
print("Problem data:", prob.get_problem_data(cp.CLARABEL)[0]['A'].toarray())
prob.solve()
print("status:", prob.status)
print("optimal value:", prob.value)
print("optimal var:", x.value)