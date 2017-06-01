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
import scipy.sparse as sp
np.random.seed(5)
n = 10000
m = 100
pbar = (np.ones((n, 1)) * .03 +
        np.matrix(np.append(np.random.rand(n - 1, 1), 0)).T * .12)

F = sp.rand(m, n, density=0.01)
F.data = np.ones(len(F.data))
D = sp.eye(n).tocoo()
D.data = np.random.randn(len(D.data))**2
# num_points=100 # number of points in each vector
# num_vects=m-1
# vals=[]
# for _ in range(num_vects):
#     vals.append(np.random.normal(size=num_points))
# vals.append(np.ones(num_points)*.03)
# Z = np.cov(vals)
Z = np.random.normal(size=(m, m))
Z = Z.T.dot(Z)
print Z.shape

x = Variable(n)
y = x.__rmul__(F)
mu = 1
ret = pbar.T * x
risk = square(norm(x.__rmul__(D))) + quad_form(y, Z)
objective = Minimize( -ret + mu * risk )

constraints_longonly = [sum_entries(x) == 1, x >= 0]

prob = Problem(objective, constraints_longonly)
#constraints_totalshort = [sum_entries(x) == 1, one.T * max(-x, 0) <= 0.5]
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

