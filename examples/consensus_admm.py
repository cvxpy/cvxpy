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
    f += (rho/2)*sum_squares(x - 1000)
    #Problem(Minimize(f)).solve()
    p = Problem(Minimize(f))
    p.solve()
    if p.status is not OPTIMAL:
        print f
        print v
        p.solve(verbose=True)
    return x.value

x = Variable(n)
prox_arg = Parameter(n)
gamma = 1.0
rho = 1.0

# Initialize x, z, u.
funcs = [sum_squares(A*x - b),
         gamma*norm(x, 1)]
ui = [np.zeros((n, 1)) for func in funcs]
z = np.zeros((n, 1))
pool = Pool(2)
for i in range(200):
    # x update.
    prox_args = [-z + u for u in ui]
    xi = map(prox, zip(funcs, prox_args))
    # z update.
    xi_ui = [x_ + u for x_, u in zip(xi, ui)]
    z = sum(xi_ui)/len(xi_ui)
    # u update.
    ui = [u + x_ - z for x_, u in zip(xi, ui)]

obj = sum_squares(A*x - b) + gamma*norm(x, 1)
prob = Problem(Minimize(obj))
result = prob.solve()
# print x.value
# print z
print "ADMM best", (sum_squares(A*z - b) + gamma*norm(z, 1)).value
print "ECOS best", obj.value
# Boolean least squares with prox.

import numpy as np
from cvxpy import *
m = 4
n = 2
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)
x = Variable(n)
obj = sum_squares(A*x - b) + sum_squares(x - 1000)
prob = Problem(Minimize(obj))
prob.solve(verbose=True)
print prob.status
c, G, h, dims, A, b = prob.get_problem_data(ECOS)