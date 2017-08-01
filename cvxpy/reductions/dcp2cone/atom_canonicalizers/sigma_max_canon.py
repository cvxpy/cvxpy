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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
import scipy.sparse as sp


def sigma_max_canon(expr, args):
    A = args[0]
    n, m = A.shape
    X = Variable((n+m, n+m), PSD=True)

    shape = expr.shape
    t = Variable(shape)
    constraints = []

    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:n, 0:n] == I_n*t
    constraints.append(X[0:n, 0:n] == Constant(sp.eye(n)) * t)

    # X[0:n, n:n+m] == A
    constraints.append(X[0:n, n:n+m] == A)

    # X[n:n+m, n:n+m] == I_m*t
    constraints.append(X[n:n+m, n:n+m] == Constant(sp.eye(m)) * t)

    return t, constraints
