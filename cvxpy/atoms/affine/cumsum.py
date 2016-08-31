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
from cvxpy.expressions.constants import Constant
import scipy.sparse as sp

def get_cumsum_mat(dim):
    """Return a sparse matrix representation of cumsum operator.

    Parameters
    ----------
    dim : int
       The length of the matrix dimensions.

    Returns
    -------
    SciPy CSC matrix
        A square matrix representing cumsum.
    """
    # Construct a sparse matrix representation.
    val_arr = []
    row_arr = []
    col_arr = []
    for i in range(dim):
        for j in range(i + 1):
            val_arr.append(1.)
            row_arr.append(i)
            col_arr.append(j)

    return sp.coo_matrix((val_arr, (row_arr, col_arr)),
                         (dim, dim)).tocsc()


def cumsum(expr, axis=0):
    """Cumulative sum.

    Parameters
    ----------
    expr : CVXPY expression
        The expression being summed.
    axis : int
        The axis to sum across if 2D.

    Return
    ------
    CVXPY expression
        The CVXPY expression representing the cumulative sum.
    """
    expr = Expression.cast_to_const(expr)
    if axis == 0:
        mat = get_cumsum_mat(expr.size[0])
        return Constant(mat)*expr
    else:
        mat = get_cumsum_mat(expr.size[1]).T
        return expr*Constant(mat)
