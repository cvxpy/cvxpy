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
from sparse_bool_mat import SparseBoolMat
import numpy as np

# Converts bools, BoolMats, and SparseBoolMats into a vertically
# concatenated BoolMat.
# Each arg has the form (value,size).
def vstack(values, sizes):
    matrices = []
    for value,size in zip(values, sizes):
        if isinstance(value, bool):
            mat = np.empty(size, dtype='bool')
            mat.fill(value)
            matrices.append(mat)
        elif isinstance(value, SparseBoolMat):
            mat = value.todense()
            matrices.append( mat.value )
        else: # BoolMat
            matrices.append( value.value )
    return BoolMat( np.vstack(matrices) )

# # Multiplies matrices, promoting if necessary.
# def mul(lh_mat, lh_size, rh_mat, rh_size):
#     if lh_mat is True:
#         if lh_size == (1,1):
#             return rh_mat
#         else:
#             lh_mat = BoolMat.promote(lh_mat, lh_size)
#     elif lh_mat is False:
#         return False

#     if rh_mat is True:
#         if rh_size == (1,1):
#             return lh_mat
#         else:
#             rh_mat = BoolMat.promote(rh_mat, rh_size)
#     elif rh_mat is False:
#         return False

#     return lh_mat * rh_mat

# # Returns true if any of the entries in the matrix are True.
# def any(matrix):
#     if isinstance(matrix, bool):
#         return matrix
#     else:
#         return matrix.any()

# # Transposes a matrix. Leaves scalars untouched.
# def transpose(matrix):
#     if isinstance(matrix, bool):
#         return matrix
#     else:
#         return matrix.__class__(matrix.value.T)

# # Indexes/slices into a matrix. Leaves scalars untouched.
# def index(matrix, key):
#     if isinstance(matrix, bool):
#         return matrix
#     else:
#         return matrix[key]