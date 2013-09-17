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
from bool_mat import BoolMat
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
        elif isinstance(value, BoolMat):
            matrices.append( value.value )
        else:
            mat = value.todense()
            matrices.append( mat.value )
    return BoolMat( np.vstack(matrices) )