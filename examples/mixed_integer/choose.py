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

from boolean import BoolVar
from cvxpy.constraints.affine import AffLeqConstraint, AffEqConstraint
import cvxopt

class SparseBoolVar(BoolVar):
    """ A variable with k 1's and all other entries 0. """
    def __init__(self, rows=1, cols=1, nonzeros=None, *args, **kwargs):
        self.k = nonzeros
        super(SparseBoolVar, self).__init__(rows, cols, *args, **kwargs)

    # Sets the initial z value to the expected value of each entry.
    def init_z(self):
        num_entries = float(self.size[0]*self.size[1])
        self.z.value = cvxopt.matrix(num_entries/self.k, self.size, tc='d')

    # The k-largest values are set to 1. The remainder are set to 0.
    def _round(self, matrix):
        v_ind = sorted(enumerate(matrix), key=lambda v: -v[1])
        for v in v_ind[0:self.k]:
            matrix[v[0]] = 1
        for v in v_ind[self.k:]:
            matrix[v[0]] = 0
        return matrix