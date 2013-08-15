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
from cvxpy.constraints.affine import AffLeqConstraint, AffEqConstraint

class choose(NonCvxVariable):
    """ A variable with k 1's and all other entries 0. """
    def __init__(self, rows=1, cols=1, k=None, *args, **kwargs):
        self.k = k
        super(choose, self).__init__(rows, cols, *args, **kwargs)

    # The k-largest values are set to 1. The remainder are set to 0.
    def _round(self, matrix):
        v_ind = sorted(enumerate(matrix), key=lambda v: -v[1])
        for v in v_ind[0:self.k]:
            matrix[v[0]] = 1
        for v in v_ind[self.k:]:
            matrix[v[0]] = 0
        return matrix

    # Constrain all entries to be zero that correspond to
    # zeros in the matrix.
    def _fix(self, matrix):
        return [self == matrix]

    # In the relaxation, 0 <= var <= 1 and sum(var) == k.
    def constraints(self):
        return [AffLeqConstraint(0, self._objective()),
                AffLeqConstraint(self._objective(), 1),
                AffEqConstraint(sum(self), self.k)]