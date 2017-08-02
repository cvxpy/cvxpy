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

from __future__ import division
import sys

import cvxopt
import numpy as np
from pylab import *
import math

from cvxpy import *

# Taken from CVX website http://cvxr.com/cvx/examples/
# Section 6.1.2: Residual minimization with deadzone penalty
# Ported from cvx matlab to cvxpy by Misrab Faizullah-Khan
# Original comments below

# Boyd & Vandenberghe "Convex Optimization"
# Joelle Skaf - 08/17/05
#
# The penalty function approximation problem has the form:
#               minimize    sum(deadzone(Ax - b))
# where 'deadzone' is the deadzone penalty function
#               deadzone(y) = max(abs(y)-1,0)

# Input data
m = 16
n = 8
A = cvxopt.normal(m,n)
b = cvxopt.normal(m,1)

# Formulate the problem
x = Variable(n)
objective = Minimize( sum(maximum( abs(A*x -b) - 1 , 0 )) )
p = Problem(objective, [])

# Solve it
print 'Computing the optimal solution of the deadzone approximation problem:'
p.solve()

print 'Optimal vector:'
print x.value

print 'Residual vector:'
print A*x.value - b

