import pdb
from math import pi

import numpy as np
import numpy.linalg as LA

import cvxpy as cp
from cvxpy import log, square

np.random.seed(1234)
n = 2
data = 10 * np.random.randn(n)
sigma_opt = (1 / np.sqrt(n)) * LA.norm(data - np.mean(data))
mu_opt = np.mean(data)
mu = cp.Variable((1, ), name="mu")
TO_RUN = 2

# for n = 200, the first one doesn't work if we use cp.sum(cp.square(data-mu)) but it works 
# with sum_of_squares

# how is the prod canoncalized? maybe that's the issue. Hmm, or the chain rule! Start in the
# opt solution and see what happens

if TO_RUN == 1:
    # here we wont induce that sigma is nonnegative so it can be useful to mention it
    sigma = cp.Variable((1, ), nonneg=True, name="sigma")
    obj = (n / 2) * log(2*pi*square(sigma)) + (1 / (2 * square(sigma))) * cp.sum(cp.square(data-mu))
    constraints = []
elif TO_RUN == 2:
    # here we will induce that sigma2 is nonnegative so no need to mention it
    sigma2 = cp.Variable((1, ), name="sigma2")
    obj = (n / 2) * log(2*pi*sigma2) + (1 / (2 * sigma2)) * cp.sum(cp.square(data-mu))
    constraints = []
    sigma = cp.sqrt(sigma2)
elif TO_RUN == 3:
    sigma = cp.Variable((1, ))
    #sigma.value = np.array([1 * sigma_opt])
    #mu.value = np.array([1 * mu_opt])
    #t = cp.Variable((n, ))
    #v = cp.Variable((1, ), bounds=[0, None])
    obj = n  * log(np.sqrt(2*pi)*sigma) + (1 / (2 * square(sigma))) * cp.sum(cp.square(data-mu))
    #obj = n  * log(np.sqrt(2*pi)*sigma) + (1 / (2 * square(sigma))) * cp.sum_squares(data-mu)
    constraints = []

problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.IPOPT, nlp=True)

print("mu difference: ", mu.value - np.mean(data))
print("sigma difference: ", sigma.value - sigma_opt)
pdb.set_trace()
