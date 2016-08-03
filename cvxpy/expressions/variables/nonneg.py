"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy.expressions.variables.variable import Variable
import cvxpy.lin_ops.lin_utils as lu


class NonNegative(Variable):
    """A variable constrained to be nonnegative.
    """

    def canonicalize(self):
        """Enforce that var >= 0.
        """
        obj, constr = super(NonNegative, self).canonicalize()
        return (obj, constr + [lu.create_geq(obj)])

    def __repr__(self):
        return "NonNegative(%d, %d)" % self.size

    def is_positive(self):
        """Is the expression positive?
        """
        return True

    def is_negative(self):
        """Is the expression negative?
        """
        return False
