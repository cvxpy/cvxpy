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

# Utility functions for computing gradients.

import scipy.sparse as sp


def constant_grad(expr):
    """Returns the gradient of constant terms in an expression.

    Matrix expressions are vectorized, so the gradient is a matrix.

    Args:
        expr: An expression.

    Returns:
        A map of variable value to empty SciPy CSC sparse matrices.
    """
    grad = {}
    for var in expr.variables():
        rows = var.size[0]*var.size[1]
        cols = expr.size[0]*expr.size[1]
        # Scalars -> 0
        if (rows, cols) == (1, 1):
            grad[var] = 0.0
        else:
            grad[var] = sp.csc_matrix((rows, cols), dtype='float64')
    return grad


def error_grad(expr):
    """Returns a gradient of all None.

    Args:
        expr: An expression.

    Returns:
        A map of variable value to None.
    """
    return {var: None for var in expr.variables()}
