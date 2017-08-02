"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

# simple_portfolio_data
from cvxpy import *
import numpy as np
np.random.seed(5)
n = 8000
pbar = (np.ones((n, 1)) * .03 +
        np.matrix(np.append(np.random.rand(n - 1, 1), 0)).T * .12)
S = np.matrix(np.random.randn(n, n))
S = S.T * S
S = S / np.max(np.abs(np.diag(S))) * .2
S[:, n - 1] = np.matrix(np.zeros((n, 1)))
S[n - 1, :] = np.matrix(np.zeros((1, n)))
x_unif = np.matrix(np.ones((n, 1))) / n

x = Variable(n)
mu = 1
ret = pbar.T * x
risk = quad_form(x, S)
objective = Minimize( -ret + mu * risk )

constraints_longonly = [sum(x) == 1, x >= 0]

prob = Problem(objective, constraints_longonly)
#constraints_totalshort = [sum(x) == 1, one.T * max(-x, 0) <= 0.5]
import time
print "starting problems"

start = time.clock()
prob.solve(verbose=True, solver=SCS)
elapsed = (time.clock() - start)
print "SCS time:", elapsed
print prob.value

start = time.clock()
prob.solve(verbose=True, solver=ECOS)
elapsed = (time.clock() - start)
print "ECOS time:", elapsed
print prob.value

start = time.clock()
prob.solve(verbose=True, solver=CVXOPT)
elapsed = (time.clock() - start)
print "CVXOPT time:", elapsed
print prob.value

# Results:
# n = 500, total 0.647 (SCS)
# parse 0.22
# SCS 0.429, ECOS 1.806, CVXOPT 2.434 (total)
# n = 1000, total 17.03892 (ECOS)
# parse .96
# ECOS 16.079496, CVXOPT 15.485536 (w/ parse), SCS 2.09
# n = 2000, total 14.488 (SCS)
# parse 3.8
# SCS 10.7, ECOS failed after 140.4, CVXOPT 121.834
# n = 4000, total 80.56 (SCS)
# parse 15.7
# SCS 64.8
# n = 8000
# CVXOPT time: 8082.954368
# ECOS time: 12651.672262 (12587.26)
# SCS time: 351.276727 (2.84e+02s)
