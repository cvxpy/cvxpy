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
    
# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

# Serial computation.
x_values = [get_x(value) for value in np.logspace(-1, 2, num=100)]

# Parallel computation.
# Create a pool of workers and a grid of gamma values.
pool = Pool(processes = 4)
gammas = numpy.logspace(-1, 2, num=100)

# Compute allocation in parallel.
par_x = pool.map(get_x, gammas)

for v1,v2 in zip(x_values, par_x):
    if numpy.linalg.norm(v1 - v2) > 1e-5:
        print "error"