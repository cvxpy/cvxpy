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

from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Parameter
import cvxopt
import numpy as np

class Boolean(Variable):
    def __init__(self, rows=1, cols=1, *args, **kwargs):
        self._LB = Parameter(rows, cols)
        self._LB.value = cvxopt.matrix(0,(rows, cols), tc='d')
        self._UB = Parameter(rows, cols)
        self._UB.value = cvxopt.matrix(1,(rows, cols), tc='d')
        self._fix_values = cvxopt.matrix(False,(rows, cols))
        super(Boolean, self).__init__(rows, cols, *args, **kwargs)

    def round(self):
        self.LB = cvxopt.matrix(self._rounded, self.size)
        self.UB = cvxopt.matrix(self._rounded, self.size)

    def relax(self):
        # if fix_value is true, do not change LB and UB
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                if not self.fix_values[i, j]:
                    self.LB[i, j] = 0
                    self.UB[i, j] = 1

    def set(self, value):
        if not isinstance(value, bool): raise "Must set to boolean value"
        self.LB = cvxopt.matrix(value, self.size)
        self.UB = cvxopt.matrix(value, self.size)
        self.fix_values = cvxopt.matrix(True, self.size)

    def unset(self):
        self.fix_values = cvxopt.matrix(False, self.size)

    @property
    def _rounded(self):
        # WARNING: attempts to access self.value
        if self.size == (1, 1):
            return round(self.value)
        else:
            return np.around(self.value)

    @property
    def LB(self):
        return self._LB.value

    @LB.setter
    def LB(self, value):
        self._LB.value = value

    @property
    def UB(self):
        return self._UB.value

    @UB.setter
    def UB(self, value):
        self._UB.value = value

    @property
    def fix_values(self):
        return self._fix_values

    @fix_values.setter
    def fix_values(self, value):
        self._fix_values = value
