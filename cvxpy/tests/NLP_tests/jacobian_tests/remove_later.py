import numpy as np

import cvxpy as cp

n = 2
x = cp.Variable((n,), name='x')
x.value = np.array([1.0, 2.0])
expr1 = cp.log(x)[:, None]
expr2 = cp.log(x)[None, :]
expr = expr1 + expr2
result_dict = expr.jacobian()
correct_jacobian = np.zeros((n*n, n*n))
correct_jacobian[0, 0] = 1/x.value[0]
correct_jacobian[1, 1] = 1/x.value[1]
computed_jacobian = np.zeros((n*n, n*n))
rows, cols, vals = result_dict[x]
computed_jacobian[rows, cols] = vals
assert(np.allclose(computed_jacobian, correct_jacobian))
