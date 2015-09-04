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

from cvxpy import settings as s
from cvxpy import utilities as u
from cvxpy import interface as intf
from cvxpy.expressions.leaf import Leaf
import cvxpy.lin_ops.lin_utils as lu

class Parameter(Leaf):
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
        self._value = None
        if value is not None:
            self.value = value
        super(Parameter, self).__init__()

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self._rows, self._cols, self._name, self.sign_str, self.value]

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
        self._value = self._validate_value(val)

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

    def __repr__(self):
        """String to recreate the object.
        """
        return 'Parameter(%d, %d, sign="%s")' % (self._rows,
                                                 self._cols,
                                                 self.sign)
