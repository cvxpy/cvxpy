"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Tuple

import scipy.sparse as sp

import cvxpy.lin_ops.lin_utils as lu

# Utility functions for constraints.


def format_axis(t, X, axis):
    """Formats all the row/column cones for the solver.

    Parameters
    ----------
        t: The scalar part of the second-order constraint.
        X: A matrix whose rows/columns are each a cone.
        axis: Slice by column 0 or row 1.

    Returns
    -------
    list
        A list of LinLeqConstr that represent all the elementwise cones.
    """
    # Reduce to norms of columns.
    if axis == 1:
        X = lu.transpose(X)
    # Create matrices Tmat, Xmat such that Tmat*t + Xmat*X
    # gives the format for the elementwise cone constraints.
    cone_size = 1 + X.shape[0]
    terms = []
    # Make t_mat
    mat_shape = (cone_size, 1)
    t_mat = sp.csc_array(([1.0], ([0], [0])), mat_shape)
    t_mat = lu.create_const(t_mat, mat_shape, sparse=True)
    t_vec = t
    if not t.shape:
        # t is scalar
        t_vec = lu.reshape(t, (1, 1))
    else:
        # t is 1D
        t_vec = lu.reshape(t, (1, t.shape[0]))
    mul_shape = (cone_size, t_vec.shape[1])
    terms += [lu.mul_expr(t_mat, t_vec, mul_shape)]
    # Make X_mat
    if len(X.shape) == 1:
        X = lu.reshape(X, (X.shape[0], 1))
    mat_shape = (cone_size, X.shape[0])
    val_arr = (cone_size - 1)*[1.0]
    row_arr = range(1, cone_size)
    col_arr = range(cone_size-1)
    X_mat = sp.csc_array((val_arr, (row_arr, col_arr)), mat_shape)
    X_mat = lu.create_const(X_mat, mat_shape, sparse=True)
    mul_shape = (cone_size, X.shape[1])
    terms += [lu.mul_expr(X_mat, X, mul_shape)]
    return [lu.create_geq(lu.sum_expr(terms))]


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
    # Matrix spaces out columns of the LinOp expressions.
    mat_shape = (spacing*vars_[0].shape[0], vars_[0].shape[0])
    terms = []
    for i, var in enumerate(vars_):
        mat = get_spacing_matrix(mat_shape, spacing, i)
        terms.append(lu.mul_expr(mat, var))
    return [lu.create_geq(lu.sum_expr(terms))]


def get_spacing_matrix(shape: Tuple[int, ...], spacing, offset):
    """Returns a sparse matrix LinOp that spaces out an expression.

    Parameters
    ----------
    shape : tuple
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
    for var_row in range(shape[1]):
        val_arr.append(1.0)
        row_arr.append(spacing*var_row + offset)
        col_arr.append(var_row)
    mat = sp.csc_array((val_arr, (row_arr, col_arr)), shape)
    return lu.create_const(mat, shape, sparse=True)
