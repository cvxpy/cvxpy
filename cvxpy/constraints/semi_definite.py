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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.constraints.constraint import Constraint

class SDP(Constraint):
    """
    A semi-definite cone constraint:
        { symmetric A | x.T*A*x >= 0 for all x }
    (the set of all symmetric matrices such that the quadratic
    form x.T*A*x is positive for all x).

    Attributes:
        A: The matrix variable constrained to be semi-definite.
    """
    def __init__(self, A):
        self.A = A
        super(SDP, self).__init__()

    def __str__(self):
        return "SDP(%s)" % self.A

    def format(self):
        """Formats SDP constraints as inequalities for the solver.
        """
        # 0 <= A
        return [lu.create_geq(self.A)]

    @property
    def size(self):
        """The dimensions of the semi-definite cone.
        """
        return self.A.size
