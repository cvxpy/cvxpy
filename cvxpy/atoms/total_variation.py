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

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm import norm
from cvxpy.atoms.elementwise.norm2_elemwise import norm2_elemwise
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum_entries import sum_entries

def tv(value):
    """Total variation of a vector or matrix.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast_to_const(value)
    rows, cols = value.size
    if value.is_scalar():
        raise ValueError("tv cannot take a scalar argument.")
    # L1 norm for vectors.
    elif value.is_vector():
        return norm(value[1:] - value[0:max(rows, cols)-1], 1)
    # L2 norm for matrices.
    else:
        row_diff = value[0:rows-1, 1:cols] - value[0:rows-1, 0:cols-1]
        col_diff = value[1:rows, 0:cols-1] - value[0:rows-1, 0:cols-1]
        return sum_entries(norm2_elemwise(row_diff, col_diff))
