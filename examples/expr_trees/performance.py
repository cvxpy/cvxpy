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
from collections import namedtuple
import scipy.sparse as sp
n = 100
x = Variable(n*n)
# class testClass(object):
#   def __init__(self, x, y, z):
#       self.x = x
#       self.y = y
#       self.z = z
obj = Minimize(sum(exp(x)))
prob = Problem(obj)
prob.solve(verbose=True)
# import cProfile
# # Point = namedtuple('Point', ['x', 'y', 'z'])
# cProfile.run("prob.solve()")
# cProfile.run("prob.solve()")
#cProfile.run("sp.eye(n*n).tocsc()")
#cProfile.run("[sp.eye(n*n).tolil()[0:1000,:] for i in range(n)] ")
# cProfile.run("[Point(i, i, i) for i in xrange(n*n)]")

# from qcml import QCML
# import numpy as np
# import ecos
# p = QCML()
# s = """
# dimensions m n

# variable x(n)
# parameter mu(n)
# parameter gamma positive
# parameter F(n,m)
# parameter D(n,n)
# maximize (mu'*x - gamma*(square(norm(F'*x)) + square(norm(D*x))))
#     sum(x) == 1
#     x >= 0
# """
# p.parse(s)
# p.canonicalize()
# n = 1000
# m = 1000
# F = np.random.randn(n, m)
# D = np.random.randn(n, n)
# p.dims = {'m': m}
# p.codegen('python')
# socp_data = p.prob2socp({'mu':1, 'gamma':1,'F':F,'D':D}, {'n': n})
# sol = ecos.solve(**socp_data)
# my_vars = p.socp2prob(sol['x'], {'n': n})

# import cvxpy as cvx
# import numpy as np
# n = 1000
# m = 1000
# F = np.random.randn(n, m)
# D = np.random.randn(n, n)
# x = cvx.Variable(n)
# obj = cvx.sum(x + cvx.square(cvx.norm(F.T*x)) + cvx.square(cvx.norm(D*x)))
# prob = cvx.Problem(cvx.Minimize(obj), [cvx.sum(x) == 1, x >= 0])
# import cProfile
# cProfile.run("""
# prob.solve(verbose=True)
# """)
