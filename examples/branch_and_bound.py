"""
Branch and bound to solve minimum cardinality problem.

minimize ||A*x - b||^2_2
subject to x in {0, 1}^n
"""

from cvxpy import *
import numpy
try:
    from Queue import PriorityQueue
except:
    from queue import PriorityQueue

# Problem data.
m = 25
n = 20
numpy.random.seed(1)
A = numpy.matrix(numpy.random.randn(m, n))
b = numpy.matrix(numpy.random.randn(m, 1))
#b = A*numpy.random.uniform(-1, 1, size=(n, 1))

# Construct the problem.
x = Variable(n)
L = Parameter(n)
U = Parameter(n)
f = lambda x: sum_squares(A*x - b)
prob = Problem(Minimize(f(x)),
               [L <= x, x <= U])

visited = 0
best_solution = numpy.inf
best_x = 0
nodes = PriorityQueue()
nodes.put((numpy.inf, 0, -numpy.ones(n), numpy.ones(n), 0))
while not nodes.empty():
    visited += 1
    # Evaluate the node with the lowest lower bound.
    _, _, L_val, U_val, idx = nodes.get()
    L.value = L_val
    U.value = U_val
    lower_bound = prob.solve()
    upper_bound = f(numpy.sign(x.value)).value
    best_solution = min(best_solution, upper_bound)
    if upper_bound == best_solution:
        best_x = numpy.sign(x.value)
    # Add new nodes if not at a leaf and the branch cannot be pruned.
    if idx < n and lower_bound < best_solution:
        for i in [-1, 1]:
            L_val[idx] = U_val[idx] = i
            nodes.put((lower_bound, i, L_val.copy(), U_val.copy(), idx + 1))

print("Nodes visited: %s out of %s" % (visited, 2**(n+1)-1))
print("Optimal solution:", best_solution)
print(best_x)
