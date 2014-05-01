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

# Methods for SCS iterative solver.

from cvxpy.lin_ops.tree_mat import mul, tmul, sum_dicts
import numpy as np
import scipy.sparse.linalg as LA


def get_mul_funcs(constraints, dims,
                  var_offsets, var_sizes, var_length):

    def accAmul(x, y):
        # y += A*x
        rows = y.shape[0]
        var_dict = vec_to_dict(x, var_offsets, var_sizes)
        y += constr_mul(constraints, var_dict, rows)

    def accATmul(x, y):
        # y += A.T*x
        terms = constr_unpack(constraints, x)
        val_dict = constr_tmul(constraints, terms)
        y += dict_to_vec(val_dict, var_offsets,
                         var_sizes, var_length)

    return (accAmul, accATmul)

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
            # Handle scalars separately.
            if np.isscalar(value):
                vector[offset:size[0]+offset] = value
            else:
                vector[offset:size[0]+offset] =  np.squeeze(value[:, col])
            offset += size[0]
    return vector

def constr_mul(constraints, var_dict, vec_size):
    """Multiplies a vector by the matrix implied by the constraints.

    Parameters
    ----------
    constraints : list
        A list of linear constraints.
    var_dict : dict
        A dictionary mapping variable id to value.
    vec_size : int
        The length of the product vector.
    """
    product = np.zeros(vec_size)
    offset = 0
    for constr in constraints:
        result = mul(constr.expr, var_dict)
        rows, cols = constr.size
        for col in range(cols):
            # Handle scalars separately.
            if np.isscalar(result):
                product[offset:offset+rows] = result
            else:
                product[offset:offset+rows] = np.squeeze(result[:, col])
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
