import numpy as np
from cvxpy import *

m = 100
n = 75
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m, 1)
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
rho = 1.0

# Initialize x, xbar, u.
funcs = [sum_squares(A*x - b),
         gamma*norm(x, 1)]
ui = [np.zeros((n, 1)) for func in funcs]
xbar = np.zeros((n, 1))
pool = Pool(NUM_PROCS)
for i in range(50):
    xbar_prev = xbar
    # x update.
    prox_args = [xbar - u for u in ui]
    xi = pool.map(prox, zip(funcs, prox_args))
    xbar = sum(xi)/len(xi)
    # u update.
    ui = [u + x_ - xbar for x_, u in zip(xi, ui)]
    # # Residuals
    # # Primal
    # ADMM_mat = np.vstack(2*[np.eye(n)])
    # print "primal", norm(np.vstack(xi) - ADMM_mat*xbar).value
    # # Dual
    # print "dual", norm(xbar - xbar_prev).value


obj = sum(funcs)
prob = Problem(Minimize(obj))
result = prob.solve()
print "ADMM best", (sum_squares(np.dot(A, xbar) - b) + gamma*norm(xbar, 1)).value
print "ECOS best", result
# Boolean least squares with prox.
