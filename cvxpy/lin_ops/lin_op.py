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
# Multiplying an expression by a constant.
MUL = "mul"
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
# A dense matrix/vector/scalar constant.
# Data: NumPy matrix.
DENSE_CONST = "dense_const"
# A sparse matrix constant.
# Data: SciPy sparse matrix.
SPARSE_CONST = "sparse_const"
# Some function of parameters.
# Data: CVXPY expression.
PARAM = "param"
