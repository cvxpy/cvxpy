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

from cvxpy import *
import numpy as np
import cvxopt
from multiprocessing import Pool

# Problem data.
n = 10
m = 5
A = cvxopt.normal(n,m)
b = cvxopt.normal(n)
gamma = Parameter(nonneg=True)

# Construct the problem.
x = Variable(m)
objective = Minimize(sum_squares(A*x - b) + gamma*norm(x, 1))
p = Problem(objective)

# Assign a value to gamma and find the optimal x.
def get_x(gamma_value):
    gamma.value = gamma_value
    result = p.solve()
    return x.value

gammas = np.logspace(-1, 2, num=100)
# Serial computation.
x_values = [get_x(value) for value in gammas]

# Parallel computation.
pool = Pool(processes = 4)
par_x = pool.map(get_x, gammas)

for v1,v2 in zip(x_values, par_x):
    if np.linalg.norm(v1 - v2) > 1e-5:
        print "error"
