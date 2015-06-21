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
                             "size",
                             "args",
                             "data"])

# The types of linear operators.

# A variable.
# Data: var id.
VARIABLE = "variable"
# Promoting a scalar expression.
# Data: None
PROMOTE = "promote"
# Multiplying an expression by a constant.
# Data: LinOp evaluating to the left hand multiple.
MUL = "mul"
# Multiplying an expression by a constant on the right.
# Data: LinOp evaluating to the right hand multiple.
RMUL = "rmul"
# Multiplying an expression elementwise by a constant.
# Data: LinOp evaluating to the left hand multiple.
MUL_ELEM = "mul_elem"
# Dividing an expression by a scalar constant.
# Data: LinOp evaluating to the divisor.
DIV = "div"
# Summing expressions.
SUM = "sum"
# Negating an expression.
NEG = "neg"
# An index/slice into an expression.
# Data: (row slice, col slice).
INDEX = "index"
# The transpose of an expression.
# Data: None.
TRANSPOSE = "transpose"
# The sum of the entries of an expression.
# Data: None
SUM_ENTRIES = "sum_entries"
# The sum of the diagonal entries of an expression.
# Data: None
TRACE = "trace"
# An expression cast into a different shape.
# Data: None
RESHAPE = "reshape"
# Converts a vector to a diagonal matrix.
# Data: None
DIAG_VEC = "diag_vec"
# Converts the diagonal of a matrix to a vector.
# Data: None
DIAG_MAT = "diag_mat"
# Vectorized upper triangular portion of a matrix.
# Data: None
UPPER_TRI = "upper_tri"
# The 1D discrete convolution of two vectors.
# Data: LinOp evaluating to the left hand term.
CONV = "conv"
# The Kronecker product of two matrices.
# Data: LinOp evaluating to the left hand term.
KRON = "kron"
# Horizontally concatenating operators.
# Data: None
HSTACK = "hstack"
# Vertically concatenating operators.
# Data: None
VSTACK = "vstack"
# A scalar constant.
# Data: Python float.
SCALAR_CONST = "scalar_const"
# A dense matrix/vector constant.
# Data: NumPy matrix.
DENSE_CONST = "dense_const"
# A sparse matrix constant.
# Data: SciPy sparse matrix.
SPARSE_CONST = "sparse_const"
# Some function of parameters.
# Data: CVXPY expression.
PARAM = "param"
# An expression with no variables.
# Data: None
NO_OP = "no_op"
# ID in coefficients for constants.
CONSTANT_ID = "constant_id"
