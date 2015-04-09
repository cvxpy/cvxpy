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
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class Boolean(NonCvxVariable):
    """ A boolean variable. """
    # Sets the initial z value to a matrix of 0.5's.
    def init_z(self):
        self.z.value = np.zeros(self.size) + 0.5

    # All values set rounded to zero or 1.
    def _round(self, matrix):
        return np.around(matrix)

    # Constrain all entries to be the value in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, we have 0 <= var <= 1.
    def canonicalize(self):
        obj, constraints = super(Boolean, self).canonicalize()
        one = lu.create_const(1, (1, 1))
        constraints += [lu.create_geq(obj),
                        lu.create_leq(obj, one)]
        return (obj, constraints)
