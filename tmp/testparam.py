import cvxpy as cp

param = cp.Parameter(sign="positive")
x = cp.Variable()
prob = cp.Problem(cp.Maximize(x  * param), [x <= param])
param.value = 5
prob.solve()

import pdb; pdb.set_trace()
param.value = 10
prob.solve()
