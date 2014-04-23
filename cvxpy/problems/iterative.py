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
from cvxopt.misc import scale
import numpy as np
import scipy.sparse.linalg as LA

def get_kkt_solver(G_constraints, dims, A_constraints,
                   var_offsets, var_sizes, x_length):
    get_linear_op = get_linear_op_factory(G_constraints, dims, A_constraints,
                                          var_offsets, var_sizes, x_length)
    def kkt_solver(W):
        lin_op = get_linear_op(W)
        x0 = np.zeros(x_length)
        def solve(x, y, z):
            b = np.hstack([x, y, z])
            x0 = LA.minres(lin_op, b, x0)
            return x0
    return kkt_solver

def get_linear_op_factory(G_constraints, dims, A_constraints,
                          var_offsets, var_sizes, x_length, W):
    A_rows = sum(c.size[0] for c in A_constraints)
    A_size = (A_rows, x_length)
    kkt_size = A_size[1] + 2*x_length

    def get_linear_op(W):
        def kkt_mul(vector):
            """Multiplies the KKT matrix by a vector.

            The KKT matrix:
                [ H     A'   GG'   ]   [ ux ]   [ bx ]
                [ A     0    0     ] * [ uy ] = [ by ]
                [ GG    0   -W'*W  ]   [ uz ]   [ bz ]

            Parameters
            ----------
            vector : NumPy ndarray
                The vector to multiply by.

            Returns
            -------
            NumPy ndarray
                The matrix-vector product
            """
            ux = vector[0:A_size[1]]
            uy = vector[A_size[1]:A_size[0]]
            uz = vector[A_size[1]+A_size[0]:]
            # Compute the product.
            # A'*uy + G'*uz
            uy_list = constr_unpack(A_constraints, uy)
            uz_list = constr_unpack(G_constraints, uz)
            ATuy = constr_tmul(A_constraints, uy_list)
            GTuz = constr_tmul(G_constraints, uz_list)
            bx_dict = sum_dicts([ATux, GTuz])
            dict_to_vec(bx_dict, var_offsets, var_sizes, bx)
            # A*ux
            by_dict = constr_mul(A_constraints, ux)
            dict_to_vec(by_dict, var_offsets, var_sizes, by)
            # G*ux - W'*W*uz
            scaled_uz = scale(scale(uz, W), W, trans='T')
            Gux_dict = constr_mul(G_constraints, ux)
            Gux = dict_to_vec(Gux_dict, var_offsets, var_sizes, bz)
            bz -= scaled_uz

            return np.hstack([bx, by, bz])

        return LA.LinearOperator((kkt_size, kkt_size), kkt_mul)

    return get_linear_op

def constr_unpack(constraints, vector):
    """Unpacks a vector into a list of values for constraints.
    """
    values = []
    offset = 0
    for constr in constraints:
        rows, cols = constr.size
        val = np.zeros((rows, cols)
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

def dict_to_vec(val_dict, var_offsets, var_sizes, vector):
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
    for id_, value in val_dict.items():
        size = var_sizes[id_]
        offset = var_offsets[id_]
        for col in range(size[1]):
            vector[offset:size[0]+offset] = value[:, col]
            offset += size[0]

def constr_mul(constraints, var_dict):
    """Multiplies a vector by the matrix implied by the constraints.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    var_dict : dict
        A dictionary mapping variable id to value.

    Returns
    -------
    list
        The product for each constraint.
    """
    products = []
    for constr in constraints:
        products.append(mul(constr.expr, var_dict))
    return products

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
