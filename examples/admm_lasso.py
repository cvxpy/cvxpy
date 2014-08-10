import numpy as np

m = 10
n = 5
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)

from cvxpy import *
from multiprocessing import Pool

def prox(f, v):
    prox_arg.value = v
    Problem(Minimize(f)).solve()
    return prox_var.value

prox_var = Variable(n)
prox_arg = Parameter(n)
gamma = Parameter(sign="positive")
gamma.value = 1
rho = Parameter(sign="positive")
rho.value = 1

print rho.sign
print (rho/2).sign
prox_term = (rho/2)*sum_squares(prox_var - prox_arg)
f = sum_squares(A*prox_var + b) + prox_term
g = gamma*norm(prox_var, 1) + prox_term

# pool = Pool(2)
# results = pool.map()

# Initialize x, z, u.
x = z = u = np.zeros(n)
for i in range(100):
    # x update.
    x = prox(f, -z + u)
    # z update.
    z = prox(g, x + u)
    # u update.
    u += x - z
print x

# Boolean least squares with prox.
