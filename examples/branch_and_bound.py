"""
Branch and bound to solve minimum cardinality problem.

minimize ||A*x - b||^2_2
subject to x in {0, 1}^n
"""

from cvxpy import *
import numpy as np
from cvxopt import matrix
from Queue import PriorityQueue

# Problem data.
m = 30
n = 25
np.random.seed(1)
A = np.matrix(np.random.randn(m, n))
b = np.matrix(np.random.randn(m, 1))

# Construct the problem.
x = Variable(n)
L = Parameter(n)
U = Parameter(n)
f = lambda x: sum_squares(A*x - b)
prob = Problem(Minimize(f(x)),
               [L <= x, x <= U])

visited = 0
best_upper = best_lower = np.inf
leaves = PriorityQueue()
leaves.put((np.inf, 0, np.zeros(n), np.ones(n), 0))
while not leaves.empty():
    visited += 1
    # Evaluate the leaf with the lowest lower bound.
    _, _, L_val, U_val, next_split = leaves.get()
    L.value = L_val
    U.value = U_val
    new_lower = prob.solve()
    new_upper = f(np.round(x.value)).value
    # Update the best upper and lower bound.
    best_lower = min(best_lower, new_lower)
    best_upper = min(best_upper, new_upper)
    # Add new leaves if there are still indices to split
    # and the branch cannot be pruned.
    if next_split < n and new_lower < best_upper:
        for i in [0, 1]:
            next_L = L_val.copy()
            next_U = U_val.copy()
            next_L[next_split] = next_U[next_split] = i
            entry = (new_lower, i, next_L, next_U, next_split + 1)
            leaves.put(entry)

print "Leaves visited: %s out of %s" % (visited, 2**(n+1)-1)
print "Optimal solution:", best_upper
