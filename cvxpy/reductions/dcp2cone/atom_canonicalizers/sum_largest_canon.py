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

from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.expressions.variables.variable import Variable


def sum_largest_canon(expr, args):
    x = args[0]
    k = expr.k
    shape = expr.shape

    # min sum_entries(t) + kq
    # s.t. x <= t + q
    #      0 <= t
    t = Variable(*shape)
    q = Variable(1)
    obj = sum_entries(t) + k*q
    constraints = [x <= t + q, t >= 0]
    return obj, constraints
