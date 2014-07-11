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

# Utility functions for constraints.

import cvxpy.lin_ops.lin_utils as lu
import scipy.sparse as sp

def format_elemwise(vars_):
    """Formats all the elementwise cones for the solver.

    Parameters
    ----------
    vars_ : list
        A list of the LinOp expressions in the elementwise cones.

    Returns
    -------
    list
        A list of LinLeqConstr that represent all the elementwise cones.
    """
    # Create matrices Ai such that 0 <= A0*x0 + ... + An*xn
    # gives the format for the elementwise cone constraints.
    spacing = len(vars_)
    prod_size = (spacing*vars_[0].size[0], vars_[0].size[1])
    # Matrix spaces out columns of the LinOp expressions.
    mat_size = (spacing*vars_[0].size[0], vars_[0].size[0])
    terms = []
    for i, var in enumerate(vars_):
        mat = get_spacing_matrix(mat_size, spacing, i)
        terms.append(lu.mul_expr(mat, var, prod_size))
    return [lu.create_geq(lu.sum_expr(terms))]

def get_spacing_matrix(size, spacing, offset):
    """Returns a sparse matrix LinOp that spaces out an expression.

    Parameters
    ----------
    size : tuple
        (rows in matrix, columns in matrix)
    spacing : int
        The number of rows between each non-zero.
    offset : int
        The number of zero rows at the beginning of the matrix.

    Returns
    -------
    LinOp
        A sparse matrix constant LinOp.
    """
    val_arr = []
    row_arr = []
    col_arr = []
    # Selects from each column.
    for var_row in range(size[1]):
        val_arr.append(1.0)
        row_arr.append(spacing*var_row + offset)
        col_arr.append(var_row)
    mat = sp.coo_matrix((val_arr, (row_arr, col_arr)), size).tocsc()
    return lu.create_const(mat, size, sparse=True)
