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

import constant
from ... import utilities as u
from ... import interface as intf

class Parameter(constant.Constant):
    """
    A parameter, either matrix or scalar.
    """
    def __init__(self, rows=1, cols=1, name=None, sign="unknown"):
        self._rows = rows
        self._cols = cols
        self.sign_str = sign
        super(Parameter, self).__init__(None, name)

    def set_context(self):
        shape = u.Shape(self._rows, self._cols)
        sign = u.Sign.name_to_sign(self.sign_str)
        self._context = u.Context(sign, u.Curvature.CONSTANT, shape)

    # Getter and setter for parameter value.
    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        size = intf.size(val)
        if size != self.size:
            raise Exception("Invalid dimensions (%s,%s) for Parameter value." % size)
        # All signs are valid if sign is unknown.
        # Otherwise value sign must match declared sign.
        sign = intf.sign(val)
        if self.sign == u.Sign.POSITIVE and not sign.is_positive() or \
           self.sign == u.Sign.NEGATIVE and not sign.is_negative():
            raise Exception("Invalid sign for Parameter value.")
        self._value = val