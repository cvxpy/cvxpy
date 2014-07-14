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
from constant import Constant
import cvxpy.lin_ops.lin_utils as lu

class Parameter(Constant):
    """
    A parameter, either matrix or scalar.
    """
    PARAM_COUNT = 0
    def __init__(self, rows=1, cols=1, name=None, sign="unknown", value=None):
        self.id = lu.get_id()
        self._rows = rows
        self._cols = cols
        self.sign_str = sign
        if name is None:
            self._name = "%s%d" % (s.PARAM_PREFIX, self.id)
        else:
            self._name = name
        self.init_dcp_attr()
        # Initialize with value if provided.
        if value is not None:
            self.value = value

    def name(self):
        return self._name

    def init_dcp_attr(self):
        shape = u.Shape(self._rows, self._cols)
        sign = u.Sign(self.sign_str)
        self._dcp_attr = u.DCPAttr(sign, u.Curvature.CONSTANT, shape)

    # Getter and setter for parameter value.
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        # Convert val to the proper matrix type.
        val = intf.DEFAULT_INTERFACE.const_to_matrix(val)
        size = intf.size(val)
        if size != self.size:
            raise Exception(
                ("Invalid dimensions (%s, %s) for Parameter value." % size)
            )
        # All signs are valid if sign is unknown.
        # Otherwise value sign must match declared sign.
        sign = intf.sign(val)
        if self.is_positive() and not sign.is_positive() or \
           self.is_negative() and not sign.is_negative():
            raise Exception("Invalid sign for Parameter value.")
        self._value = val

    def parameters(self):
        """Returns itself as a parameter.
        """
        return [self]

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_param(self, self.size)
        return (obj, [])
