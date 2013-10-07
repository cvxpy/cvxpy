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
from ..expressions import types
from constraint import Constraint
from affine import AffEqConstraint, AffLeqConstraint

class LeqConstraint(Constraint):
    OP_NAME = "<="
    interface = intf.DEFAULT_INTERFACE
    @property
    def size(self):
        return self._expr.size

    # The value of the dual variable.
    @property
    def dual(self):
        return self.dual_value

    # Left hand expression must be convex and right hand must be concave.
    def is_dcp(self):
        return self._expr.curvature.is_convex()

    # Replace inequality with an equality with slack.
    def canonicalize(self):
        self._expr = (self.lh_exp - self.rh_exp)
        obj,constr = self._expr.canonical_form()
        dual_holder = AffLeqConstraint(obj, 0, self)
        return (None, [dual_holder] + constr)