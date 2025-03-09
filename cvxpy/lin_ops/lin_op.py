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

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
"""
from typing import Tuple


# A linear operator applied to a variable
# or a constant or function of parameters.
class LinOp:
    def __init__(self, type, shape: Tuple[int, ...], args, data) -> None:
        self.type = type
        self.shape = shape
        self.args = args
        self.data = data  # This is later set to C++ LinOp objects' linOp_data_ field.

    def __repr__(self) -> str:
        return f"LinOp({self.type}, {self.shape})"


# The types of linear operators.

# A variable.
# Data: var id.
VARIABLE = "variable"
# Promoting a scalar expression.
# Data: None
PROMOTE = "promote"
# Broadcasting an expression.
# Data: Shape to broadcast to.
BROADCAST_TO = "broadcast_to"
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
# Data: int, diagonal offset
DIAG_VEC = "diag_vec"
# Converts the diagonal of a matrix to a vector.
# Data: int, diagonal offset
DIAG_MAT = "diag_mat"
# Vectorized upper triangular portion of a matrix.
# Data: None
UPPER_TRI = "upper_tri"
# The 1D discrete convolution of two vectors.
# Data: LinOp evaluating to the left hand term.
CONV = "conv"
# The Kronecker product of two matrices.
# Data: LinOp evaluating to the left hand term (variable in the right-hand term).
KRON_R = "kron_r"
# Data: LinOp evaluating to the right hand term (variable in the left-hand term).
KRON_L = "kron_l"
# Horizontally concatenating operators.
# Data: None
HSTACK = "hstack"
# Vertically concatenating operators.
# Data: None
VSTACK = "vstack"
# Stack concatenating operators.
# Data: None
CONCATENATE = "concatenate"
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
CONSTANT_ID = -1
