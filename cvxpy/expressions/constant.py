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

import expression
import leaf
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
import cvxpy.utilities as u

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

class IndexConstant(Constant):
    """ An index into a matrix constant """
    # parent - the constant indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        self._shape = u.Shape(1,1)
        super(IndexConstant, self).__init__(
            intf.index(self.parent.value, self.key)
        )