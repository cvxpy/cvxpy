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

import abc
import cvxpy
import cvxpy.interface as intf
import cvxopt

class NonCvxVariable(cvxpy.Variable):
    __metaclass__ = abc.ABCMeta
    def __init__(self, *args, **kwargs):
        super(NonCvxVariable, self).__init__(*args, **kwargs)
        self.noncvx = True
        self.z = cvxpy.Parameter(*self.size)
        self.init_z()
        self.u = cvxpy.Parameter(*self.size)
        self.u.value = cvxopt.matrix(0, self.size, tc='d')

    # Initializes the value of the replicant variable.
    def init_z(self):
        self.z.value = cvxopt.matrix(0, self.size, tc='d')

    # Verify that the matrix has the same dimensions as the variable.
    def validate_matrix(self, matrix):
        if self.size != intf.size(matrix):
            raise Exception(("The argument's dimensions must match "
                             "the variable's dimensions."))

    # Wrapper to validate matrix.
    def round(self, matrix):
        self.validate_matrix(matrix)
        return self._round(matrix)

    # Project the matrix into the space defined by the non-convex constraint.
    # Returns the updated matrix.
    @abc.abstractmethod
    def _round(matrix):
        return NotImplemented

    # Wrapper to validate matrix and update curvature.
    def fix(self, matrix):
        matrix = self.round(matrix)
        return self._fix(matrix)

    # Fix the variable so it obeys the non-convex constraint.
    @abc.abstractmethod
    def _fix(self, matrix):
        return NotImplemented
