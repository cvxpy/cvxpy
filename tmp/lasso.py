import cvxpy as cp
import numpy
import time

# Problem data.
n = 15
m = 10
numpy.random.seed(1)
A = numpy.random.randn(n, m)
b = numpy.random.randn(n)
# gamma must be nonnegative due to DCP rules.
gamma = cp.Parameter(nonneg=True)

# Construct the problem.
x = cp.Variable(m)
error = cp.sum_squares(A*x - b)
obj = cp.Minimize(error + gamma*cp.norm(x, 1))
prob = cp.Problem(obj)

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
gamma_vals = numpy.logspace(-4, 6)
for val in gamma_vals:
    gamma.value = val
    start = time.time()
    prob.solve(cp.SCS)
    end = time.time()
    new_prob = cp.Problem(obj)
    print('time: ', end - start)

    start = time.time()
    new_prob.solve(cp.SCS)
    end = time.time()
    print('new prob time: ', end - start)
