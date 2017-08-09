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

import cvxpy.interface as intf
from cvxpy.expressions.leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu


class Constant(Leaf):
    """
    A constant value.

    Raw numerical constants (Python primite types, NumPy ndarrays,
    and NumPy matrices) are implicitly cast to constants via Expression
    operator overloading. For example, if ``x`` is an expression and
    ``c`` is a raw constant, then ``x + c`` creates an expression by
    casting ``c`` to a Constant.
    """

    def __init__(self, value):
        # Keep sparse matrices sparse.
        if intf.is_sparse(value):
            self._value = intf.DEFAULT_SPARSE_INTF.const_to_matrix(
                value, convert_scalars=True)
            self._sparse = True
        else:
            self._value = intf.DEFAULT_INTF.const_to_matrix(value)
            self._sparse = False
        # Set DCP attributes.
        is_nonneg, is_nonpos = intf.sign(self.value)
        super(Constant, self).__init__(intf.shape(self.value), nonneg=is_nonneg, nonpos=is_nonpos)

    def name(self):
        """The value as a string.
        """
        return str(self.value)

    def constants(self):
        """Returns self as a constant.
        """
        return [self]

    @property
    def value(self):
        """NumPy.ndarray or None: The numeric value of the constant.
        """
        return self._value

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {}

    @property
    def shape(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return self._shape

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_const(self.value, self.shape, self._sparse)
        return (obj, [])

    def __repr__(self):
        """Returns a string with information about the expression.
        """
        return "Constant(%s, %s, %s)" % (self.curvature,
                                         self.sign,
                                         self.shape)
