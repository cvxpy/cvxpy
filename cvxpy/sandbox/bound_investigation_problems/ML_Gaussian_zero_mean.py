from math import pi

import numpy as np
import numpy.linalg as LA

import cvxpy as cp
from cvxpy import log, square

np.random.seed(1234)
n = 10
data = np.random.randn(n)
sigma_opt = (1 / np.sqrt(n)) * LA.norm(data)
res = LA.norm(data) ** 2

TO_RUN = 2

if TO_RUN == 1:
    sigma = cp.Variable((1, ))
    obj = (n / 2) * log(2*pi*square(sigma)) + (1 / (2 * square(sigma))) * res
    constraints = []
elif TO_RUN == 2:
    sigma2 = cp.Variable((1, ))
    obj = (n / 2) * log(2*pi*sigma2) + (1 / (2 * sigma2)) * res
    constraints = []
    sigma = cp.sqrt(sigma2)
elif TO_RUN == 3:
    sigma = cp.Variable((1, ))
    obj = n  * log(np.sqrt(2*pi)*sigma) + (1 / (2 * square(sigma))) * res
    constraints = []

problem = cp.Problem(cp.Minimize(obj), constraints)
problem.solve(solver=cp.IPOPT, nlp=True)
print("difference sigma:", sigma.value - sigma_opt)

