"""
Branch and bound to solve minimum cardinality problem.

minimize ||A*x - b||^2_2
subject to x in {0, 1}^n
"""

from cvxpy import *
import numpy as np
from heapq import *

# Problem data.
m = 50
n = 25
np.random.seed(1)
A = np.matrix(np.random.randn(m, n))
b = A*np.random.uniform(0, 1, size=(n, 1))

# Construct the problem.
x = Variable(n)
L = Parameter(n)
U = Parameter(n)
f = lambda x: sum_squares(A*x - b)
prob = Problem(Minimize(f(x)),
               [L <= x, x <= U])

visited = 0
best_solution = np.inf
best_x = 0
nodes = []
heappush(nodes, (np.inf, 0, np.zeros(n), np.ones(n), 0))
while nodes:
    visited += 1
    # Evaluate the node with the lowest lower bound.
    _, _, L_val, U_val, idx = heappop(nodes)
    L.value = L_val
    U.value = U_val
    lower_bound = prob.solve()
    upper_bound = f(np.round(x.value)).value
    best_solution = min(best_solution, upper_bound)
    if upper_bound == best_solution:
        best_x = np.round(x.value)
    # Add new nodes if not at a leaf and the branch cannot be pruned.
    if idx < n and lower_bound < best_solution:
        for i in [0, 1]:
            L_val[idx] = U_val[idx] = i
            heappush(nodes, (lower_bound, i, L_val.copy(), U_val.copy(), idx + 1))

print("Nodes visited: %s out of %s" % (visited, 2**(n+1)-1))
print("Optimal solution:", best_solution)
print("Total non-zeros:", best_x.sum())
