import numpy as np
from cvxpy import *

m = 4
n = 2
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)
# # Precondition A and b.
# for row in range(m):
#     A[row, :] /= norm(A[row, :]).value
# for col in range(n):
#     A[:, col] /= norm(A[:, col]).value

# b /= norm(b).value

from cvxpy import *
from multiprocessing import Pool

def prox(args):
    f, v = args
    f += (rho/2)*sum_squares(x - v)
    Problem(Minimize(f)).solve()
    return x.value

x = Variable(n)
gamma = 1.0
rho = 5.0

# Initialize x, z, u.
funcs = [sum_squares(A*x - b),
         gamma*norm(x, 1)]
ui = [np.zeros((n, 1)) for func in funcs]
xbar = np.zeros((n, 1))
pool = Pool(2)
for i in range(200):
    # x update.
    prox_args = [xbar - u for u in ui]
    xi = map(prox, zip(funcs, prox_args))
    xbar = sum(xi)/len(xi)
    # u update.
    ui = [u + x_ - xbar for x_, u in zip(xi, ui)]

obj = sum_squares(A*x - b) + gamma*norm(x, 1)
prob = Problem(Minimize(obj))
result = prob.solve()
# print x.value
# print xbar
print "ADMM best", (sum_squares(A*xbar - b) + gamma*norm(xbar, 1)).value
print "ECOS best", obj.value
# Boolean least squares with prox.
