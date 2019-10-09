import cvxpy as cp
import numpy as np

x = cp.Variable(2,)
t = cp.Variable(1,)

x.value = np.ones(2)
t.value = np.ones(1)*1

quad_over_lin = lambda x, t: cp.perspective(x, t, atom=cp.sum_squares)
objective = quad_over_lin(x, t)

prob = cp.Problem(cp.Minimize(objective), [x >= 1.0, t <= 1.0])
result = prob.solve(solver=cp.SCS)

print (result)
print (objective.value)
print (x.value)
print (t.value)
