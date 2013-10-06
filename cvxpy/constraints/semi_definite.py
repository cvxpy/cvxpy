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

from .. import interface as intf
from .. import utilities as u
from ..expressions.variables import Variable
from affine import AffEqConstraint, AffLeqConstraint

class SDP(object):
    """
    A semi-definite cone constraint:
        { symmetric A | x.T*A*x >= 0 for all x }
    (the set of all symmetric matrices such that the quadratic
    form x.T*A*x is positive for all x).
    """
    # A - an affine expression or objective.
    def __init__(self, A):
        self.A = u.Affine.cast_as_affine(A)
        super(SDP, self).__init__()

    # Formats SDP constraints for the solver.
    def format(self):
        return [AffLeqConstraint(-self.A, 0)]

    # The dimensions of the semi-definite cone.
    @property
    def size(self):
        return self.A.size
