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
from affine import AffLeqConstraint

class SOC(object):
    """ 
    A second-order cone constraint:
        norm2(x) <= t
    """
    # x - an affine expression or objective.
    # t - an affine expression or objective.
    def __init__(self, t, x):
        self.x = x
        self.t = t
        super(SOC, self).__init__()

    # Formats SOC constraints for the solver.
    def format(self):
        return [AffLeqConstraint(-self.t, 0), 
                AffLeqConstraint(-self.x, 0)]

    # The dimensions of the second-order cone.
    @property
    def size(self):
        return (self.x.size[0] + self.t.size[0],1)