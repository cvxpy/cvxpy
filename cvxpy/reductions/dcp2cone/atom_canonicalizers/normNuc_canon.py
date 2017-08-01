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

from cvxpy.atoms.affine.trace import trace
from cvxpy.expressions.variable import Variable


def normNuc_canon(expr, args):
    A = args[0]
    m, n = A.shape

    # Create the equivalent problem:
    #   minimize (trace(U) + trace(V))/2
    #   subject to:
    #            [U A; A.T V] is positive semidefinite
    X = Variable((m+n, m+n), PSD=True)
    constraints = []

    # Fix X using the fact that A must be affine by the DCP rules.
    # X[0:rows,rows:rows+cols] == A
    constraints.append(X[0:m, m:m+n] == A)
    trace_value = 0.5 * trace(X)

    return trace_value, constraints
