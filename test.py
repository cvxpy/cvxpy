"""Minimum working example to generate glp_add_cols issue with an integer linear program in CVXPY.
"""

import cvxpy as cp
import numpy as np

n = 4
x = cp.Variable(n, integer=True)

A = np.array(
    [[ 52., 102.,   0.,   0.],
     [  0.,   0.,  52., 102.]])
a_max = np.array([300., 40.])
B = np.array(
    [[150., 150., 150., 150.],
     [  0., 150.,   0., 150.]])
b = np.array([300., 150.])
c = np.ones(n)

constraint_positive = x >= 0
constraint_A = A @ x <= a_max
constraint_B_low = B @ x >= 0.98 * b
constraint_B_high = B @ x <= 1.02 * b

objective = cp.Minimize(c @ x)
problem = cp.Problem(
    objective,
    [constraint_positive,
     constraint_A,
     constraint_B_low,
     constraint_B_high,
     ])

# With SCIP, the problem is solved with the correct result.
# problem.solve(solver='SCIP')

# With GLPK_MI, the problem fails with:
#   glp_add_cols: ncs = 0; invalid number of columns
#   Error detected in file api/prob1.c at line 362
#   Aborted (core dumped)
problem.solve(solver='GLPK_MI', verbose=True)

print(problem.status)
print('\nproblem.value, should be 2')
print(problem.value)
print('\nx.value, should be [1, 1, 0, 0]')
print(x.value)
