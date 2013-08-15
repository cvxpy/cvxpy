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
import cvxpy.utilities as u

class Parameter(constant.Constant):
    """
    A parameter, either matrix or scalar.
    """
    def __init__(self, rows=1, cols=1, name=None, sign="unknown"):
        self._rows = rows
        self._cols = cols
        self.sign_str = sign
        super(Parameter, self).__init__(None, name)

    # The constant's shape is fixed.
    def set_shape(self):
        self._shape = u.Shape(self._rows, self._cols)

    def set_sign(self):
        self._sign = u.Sign(self.sign_str)