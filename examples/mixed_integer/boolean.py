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

from noncvx_variable import NonCvxVariable
from cvxpy.constraints.affine import AffLeqConstraint

class boolean(NonCvxVariable):
    """ A boolean variable. """
    # All values set rounded to zero or 1.
    def _round(self, matrix):
        for i,v in enumerate(matrix):
            matrix[i] = 0 if v < 0.5 else 1
        return matrix

    # Constrain all entries to be the value in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, we have 0 <= var <= 1.
    def _constraints(self):
        return [AffLeqConstraint(0, self._objective()),
                AffLeqConstraint(self._objective(), 1)]