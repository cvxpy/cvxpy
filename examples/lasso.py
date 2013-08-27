from cvxpy import *
from cvxpy import numpy as np
import cvxopt
from multiprocessing import Pool

# Problem data.
cvxopt.setseed(1)
n = 10
m = 5
A = cvxopt.normal(n,m)
b = cvxopt.normal(n)

# Construct the problem.
gamma = Parameter(sign="positive")
x = Variable(m)
objective = Minimize(sum(square(A*x - b)) + gamma*norm1(x))
p = Problem(objective)

# Vary gamma for trade-off curve.
x_values = []
for value in np.logspace(-1, 2, num=100):
    gamma.value = value
    p.solve()
    x_values.append(x.value)
    
# Construct a trade off curve using the x_values.
# encapsulate the allocation function
def allocate(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

# Create a pool of workers and a grid of gamma values.
pool = Pool(processes = 4)
gammas = numpy.logspace(-1, 2, num=100)

# compute allocation in parallel
par_x = pool.map(allocate, gammas)
for v1,v2 in zip(x_values, par_x):
    if numpy.linalg.norm(v1 - v2) > 1e-5:
        print "error"