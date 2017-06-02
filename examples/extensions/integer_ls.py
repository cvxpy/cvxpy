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
from ncvx.boolean import Boolean
import ncvx.branch_and_bound
import cvxopt

x = Boolean(3, name='x')
A = cvxopt.matrix([1,2,3,4,5,6,7,8,9], (3, 3), tc='d')
z = cvxopt.matrix([3, 7, 9])

p = Problem(Minimize(sum_squares(A*x - z))).solve(method="branch and bound")

print x.value
print p

# even a simple problem like this introduces too many variables
# y = Boolean()
# Problem(Minimize(square(y - 0.5))).branch_and_bound()
