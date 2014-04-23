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

# An iterative KKT solver for CVXOPT.

from cvxpy.lin_ops.tree_mat import mul, tmul, sum_dicts
import cvxopt
from cvxopt.misc import scale
import numpy as np
import scipy.sparse.linalg as LA

# Regularization constant.
REG_EPS = 1e-9

def get_kktsolver(A_constraints, G_constraints, dims,
                  var_offsets, var_sizes, x_length):
    get_linear_op = get_linear_op_factory(A_constraints, G_constraints, dims,
                                          var_offsets, var_sizes, x_length)
    def kkt_solver(W):
        lin_op = get_linear_op(W)
        x0 = np.zeros(lin_op.shape[0])
        def solve(x, y, z):
            # Cast x, y, z as 1D arrays.
            scale(z, W, trans='T', inverse='I')
            x, y, z = map(lambda mat: np.asarray(mat)[:, 0], [x, y, z])
            b = np.hstack([x, y, z])
            solution, info = LA.gmres(lin_op, b, x0)
            print info
            x0[:] = solution[:]
            x[:] = solution[0:len(x)]
            y[:] = solution[len(x):len(x)+len(y)]
            z[:] = solution[len(x)+len(y):]
        return solve
    return kkt_solver

def get_linear_op_factory(A_constraints, G_constraints, dims,
                          var_offsets, var_sizes, x_length):
    A_rows = dims["f"]
    A_cols = x_length
    G_rows = dims["l"] + sum(dims["q"]) + sum(dims["s"])
    G_cols = x_length
    kkt_size = A_cols + A_rows + G_rows

    def get_linear_op(W):
        def kkt_mul(vector):
            """Multiplies the KKT matrix by a vector.

            The KKT matrix:
                [ 0          A'   GG'*W^{-1} ]   [ ux   ]   [ bx        ]
                [ A          0    0          ] * [ uy   [ = [ by        ]
                [ W^{-T}*GG  0   -I          ]   [ W*uz ]   [ W^{-T}*bz ]

            Parameters
            ----------
            vector : NumPy ndarray
                The vector to multiply by.

            Returns
            -------
            NumPy ndarray
                The matrix-vector product
            """
            ux = vector[0:A_cols]
            uy = vector[A_cols:A_cols+A_rows]
            uz = vector[A_cols+A_rows:]
            # Compute the product.
            # by = A*ux - eps*I*uy
            by = constr_mul(A_constraints, ux, A_rows)
            #by -= uy*REG_EPS
            # bz = W^{-T}*G*ux - (1+epsilon)*I*uz
            Gux = constr_mul(G_constraints, ux, G_rows)
            scale(cvxopt.matrix(Gux), W, trans='T', inverse='I')
            bz = Gux - uz #(1+REG_EPS)*uz
            # eps*I*ux + A'*uy + G'W^{-1}*uz
            uy_list = constr_unpack(A_constraints, uy)
            ATuy = constr_tmul(A_constraints, uy_list)

            scale(cvxopt.matrix(uz), W, inverse='I')
            uz_list = constr_unpack(G_constraints, uz)
            GTuz = constr_tmul(G_constraints, uz_list)

            bx_dict = sum_dicts([ATuy, GTuz])
            bx = dict_to_vec(bx_dict, var_offsets, var_sizes, x_length)
            #bx += ux*REG_EPS

            return np.hstack([bx, by, bz])

        return LA.LinearOperator((kkt_size, kkt_size), kkt_mul,
                                 dtype="float64")

    return get_linear_op

def constr_unpack(constraints, vector):
    """Unpacks a vector into a list of values for constraints.
    """
    values = []
    offset = 0
    for constr in constraints:
        rows, cols = constr.size
        val = np.zeros((rows, cols))
        for col in range(cols):
            val[:, col] = vector[offset:offset+rows]
            offset += rows
        values.append(val)
    return values

def vec_to_dict(vector, var_offsets, var_sizes):
    """Converts a vector to a map of variable id to value.

    Parameters
    ----------
    vector : NumPy matrix
        The vector of values.
    var_offsets : dict
        A map of variable id to offset in the vector.
    var_sizes : dict
        A map of variable id to variable size.

    Returns
    -------
    dict
        A map of variable id to variable value.
    """
    val_dict = {}
    for id_, offset in var_offsets.items():
        size = var_sizes[id_]
        value = np.zeros(size)
        offset = var_offsets[id_]
        for col in range(size[1]):
            value[:, col] = vector[offset:size[0]+offset]
            offset += size[0]
        val_dict[id_] = value
    return val_dict

def dict_to_vec(val_dict, var_offsets, var_sizes, vec_len):
    """Converts a map of variable id to value to a vector.

    Parameters
    ----------
    val_dict : dict
        A map of variable id to value.
    var_offsets : dict
        A map of variable id to offset in the vector.
    var_sizes : dict
        A map of variable id to variable size.
    vector : NumPy matrix
        The vector to store the values in.
    """
    # TODO take in vector.
    vector = np.zeros(vec_len)
    for id_, value in val_dict.items():
        size = var_sizes[id_]
        offset = var_offsets[id_]
        for col in range(size[1]):
            vector[offset:size[0]+offset] = value[:, col]
            offset += size[0]
    return vector

def constr_mul(constraints, var_dict, rows):
    """Multiplies a vector by the matrix implied by the constraints.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    var_dict : dict
        A dictionary mapping variable id to value.

    Returns
    -------
    NumPy 1D array
        The product for the constraints.
    """
    product = np.zeros(rows)
    offset = 0
    for constr in constraints:
        result = mul(constr.expr, var_dict)
        rows, cols = constr.size
        for col in range(cols):
            # Handle scalars separately.
            if np.isscalar(result):
                product[offset:offset+rows] = result
            else:
                product[offset:offset+rows] = result[:, col]
            offset += rows
    return product

def constr_tmul(constraints, values):
    """Multiplies a vector by the transpose of the constraints matrix.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    values : list
        A list of NumPy matrices.

    Returns
    -------
    dict
        A mapping of variable id to value.
    """
    products = []
    for constr, val in zip(constraints, values):
        products.append(tmul(constr.expr, val))
    return sum_dicts(products)
