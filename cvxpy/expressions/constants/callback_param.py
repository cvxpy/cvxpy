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

from cvxpy.expressions.constants.parameter import Parameter

class CallbackParam(Parameter):
    """
    A parameter whose value is obtained by evaluating a function.
    """
    PARAM_COUNT = 0
    def __init__(self, callback, rows=1, cols=1, name=None, sign="unknown"):
        self._callback = callback
        super(CallbackParam, self).__init__(rows, cols, name, sign)

    @property
    def value(self):
        """Evaluate the callback to get the value.
        """
        return self._validate_value(self._callback())
