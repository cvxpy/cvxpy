from __future__ import division

import numpy as np
from mixed_integer import *
from ncvx import Boolean

from cvxpy import ECOS, GUROBI, Assign, Minimize, Problem, norm, vstack

# Traveling salesman problem.
n = 10

# Get locations.
np.random.seed(1)
x = np.random.uniform(-100, 100, size=(n,1))
y = np.random.uniform(-100, 100, size=(n,1))

MAX_ITER = 50
RESTARTS = 5

# Make objective.
assignment = Assign(n, n)
ordered_x = assignment*x
ordered_y = assignment*y
cost = 0
for i in range(n-1):
    x_diff = ordered_x[i+1] - ordered_x[i]
    y_diff = ordered_y[i+1] - ordered_y[i]
    cost += norm(vstack([x_diff, y_diff]))
prob = Problem(Minimize(cost))
result = prob.solve(method="admm", iterations=MAX_ITER,
                    solver=ECOS, verbose=False)#, tau=1.1, tau_max=100)
print("all constraints hold:", np.all([c.value for c in prob.constraints]))
print("final value", result)

# print prob.solve(method="polish")
# print np.around(positions.value)

assignment = Boolean(n, n)
ordered_x = assignment*x
ordered_y = assignment*y
cost = 0
for i in range(n-1):
    x_diff = ordered_x[i+1] - ordered_x[i]
    y_diff = ordered_y[i+1] - ordered_y[i]
    cost += norm(vstack([x_diff, y_diff]))
prob = Problem(Minimize(cost),
        [assignment*np.ones((n, 1)) == 1,
         np.ones((1, n))*assignment == 1])
prob.solve(solver=GUROBI, verbose=False, TimeLimit=10)
print("gurobi solution", prob.value)
# print positions.value

# Randomly guess permutations.
total = 0
best = np.inf
for k in range(RESTARTS*MAX_ITER):
    assignment.value = np.zeros(assignment.size)
    for i, match in enumerate(np.random.permutation(n)):
        assignment.value[match, i] = 1
    if cost.value < result:
        total += 1
    if cost.value < best:
        best = cost.value
    # print positions.value
print("%% better = ", (total/(RESTARTS*MAX_ITER)))
print("best = ", best)
