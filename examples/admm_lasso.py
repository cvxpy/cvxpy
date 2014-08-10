import numpy as np
from cvxpy import *

m = 10
n = 5
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)
# Precondition A and b.
for row in range(m):
    A[row, :] /= norm(A[row, :]).value
for col in range(n):
    A[:, col] /= norm(A[:, col]).value

b /= norm(b).value

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
rho.value = 10

prox_term = (rho/2)*sum_squares(prox_var - prox_arg)
f = sum_squares(A*prox_var + b) + prox_term
g = gamma*norm(prox_var, 1) + prox_term

# pool = Pool(2)
# results = pool.map()

# Initialize x, z, u.
x = z = u = np.zeros((n, 1))
for i in range(100):
    # x update.
    x = prox(f, -z + u)
    # z update.
    z = prox(g, x + u)
    # u update.
    u += x - z
x_admm = x

obj = sum_squares(A*prox_var + b) + gamma*norm(prox_var, 1)
prob = Problem(Minimize(obj))
prob.solve()
print prox_var.value
print x_admm
print norm(prox_var - x_admm).value
# Boolean least squares with prox.
