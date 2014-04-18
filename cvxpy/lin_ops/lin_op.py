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

from collections import namedtuple

# A linear operator applied to a variable
# or a constant or function of parameters.
LinOp = namedtuple("LinOp", ["type",
                             "var_id",
                             "var_size",
                             "scalar_coeff",
                             "data"])

# The types of linear operators.

# An identity matrix multiplied by a variable.
# Data: None.
EYE_MUL = "eye_mul"
# A dense matrix/vector multiplied by a variable.
# Data: NumPy matrix.
DENSE_MUL = "dense_mul"
# A sparse matrix multiplied by a variable.
# Data: SciPy sparse matrix.
SPARSE_MUL = "sparse_mul"
# Some function of parameters multiplied by a variable.
# Data: CVXPY expression.
PARAM_MUL = "param_mul"
# An index/slice into a variable.
# Data: (row slice, col slice).
INDEX = "index"
# The transpose of a variable.
# Data: None.
TRANSPOSE = "transpose"
# The sum of the entries of a variable.
# Data: None
SUM_ENTRIES = "sum_entries"
# A scalar constant.
# Data: Python float.
SCALAR_CONST = "scalar"
# A dense matrix/vector constant.
# Data: NumPy matrix.
DENSE_CONST = "dense_const"
# A sparse matrix constant.
# Data: SciPy sparse matrix.
SPARSE_CONST = "sparse_const"
# Some function of parameters.
# Data: CVXPY expression.
PARAM = "param"

# ID for all constants.
CONSTANT_ID = "constant"

# Maps constant types by term types to the type of the product.
# Scalar constants are a special case.
MUL_TYPE = {
    # Dense
    (DENSE_CONST, EYE_MUL): DENSE_MUL,
    (DENSE_CONST, DENSE_MUL): DENSE_MUL,
    (DENSE_CONST, SPARSE_MUL): DENSE_MUL,
    (DENSE_CONST, SCALAR_CONST): DENSE_CONST,
    (DENSE_CONST, DENSE_CONST): DENSE_CONST,
    (DENSE_CONST, SPARSE_CONST): DENSE_CONST,
    # Sparse
    (SPARSE_CONST, EYE_MUL): SPARSE_MUL,
    (SPARSE_CONST, DENSE_MUL): DENSE_MUL,
    (SPARSE_CONST, SPARSE_MUL): SPARSE_MUL,
    (SPARSE_CONST, SCALAR_CONST): SPARSE_CONST,
    (SPARSE_CONST, DENSE_CONST): DENSE_CONST,
    (SPARSE_CONST, SPARSE_CONST): SPARSE_CONST,
    # Param
    (PARAM, EYE_MUL): PARAM_MUL,
}
