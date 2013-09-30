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

from ... import settings as s
from ... import utilities as u
from ... import interface as intf
from .. import expression
from .. import leaf

class Constant(leaf.Leaf):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value, name=None):
        self.value = value
        self.param_name = name
        self.set_context()
        super(Constant, self).__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def name(self):
        return str(self.value) if self.param_name is None else self.param_name

    # The constant's sign and shape are fixed.
    def set_context(self):
        shape = u.Shape(*intf.size(self.value))
        sign = intf.sign(self.value)
        self._context = u.Context(sign, u.Curvature.CONSTANT, shape)

    # Return the constant value, converted to the target matrix.
    def coefficients(self, interface):
        return {s.CONSTANT: interface.const_to_matrix(self.value)}

    # No variables.
    def variables(self):
        return []

    # Return a scalar view into a matrix constant.
    def index_object(self, key):
        return IndexConstant(self, key)

    # The transpose of the constant.
    @property
    def T(self):
        if self.size == (1,1): # Transpose of a scalar is that scalar.
            return self
        else:
            transpose_val = intf.transpose(self.value)
            return Constant(transpose_val)

    # Vectorizes the coefficient and adds it to the constant vector.
    # matrix - the constant vector.
    # coeff - the constant coefficient.
    # vert_offset - the current vertical offset.
    # constraint - the constraint containing the variable.
    # var_offsets - a map of variable object to horizontal offset.
    # interface - the interface for the matrix type.
    @classmethod
    def place_coeff(cls, matrix, coeff, vert_offset, 
                    constraint, var_offsets, interface):
        rows = constraint.size[0]*constraint.size[1]
        interface.block_add(matrix, coeff, vert_offset, 0, rows, 1)

class IndexConstant(Constant):
    """ An index into a matrix constant. """
    # parent - the constant indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        self._shape = u.Shape(1,1)
        super(IndexConstant, self).__init__(
            intf.index(self.parent.value, self.key)
        )