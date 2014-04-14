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

from .. import utilities as u

class LeqConstraint(u.Affine, u.Canonical):
    OP_NAME = "<="
    TOLERANCE = 1e-4
    def __init__(self, lh_exp, rh_exp, parent=None):
        self.lh_exp = lh_exp
        self.rh_exp = rh_exp
        self._expr = self.lh_exp - self.rh_exp
        self.parent = parent
        self._dual_value = None

    def name(self):
        return ' '.join([str(self.lh_exp),
                         self.OP_NAME,
                         str(self.rh_exp)])

    def __repr__(self):
        return self.name()

    @property
    def size(self):
        return self._expr.size

    # Left hand expression must be convex and right hand must be concave.
    def is_dcp(self):
        return self._expr.is_convex()

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the LeqConstraint.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self._expr.canonical_form
        dual_holder = self.__class__(obj, 0, parent=self)
        return (None, constraints + [dual_holder])

    def variables(self):
        """Returns the variables in the compared expressions.
        """
        return self._expr.variables()

    def parameters(self):
        """Returns the parameters in the compared expressions.
        """
        return self._expr.parameters()

    def _tree_to_coeffs(self):
        return self._expr.coefficients()

    @property
    def value(self):
        """Does the constraint hold?

        Returns
        -------
        bool
        """
        if self._expr.value is None:
            return None
        else:
            return self._expr.value <= self.TOLERANCE

    # The value of the dual variable.
    @property
    def dual_value(self):
        return self._dual_value

    def save_value(self, value):
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        if self.parent is None:
            self._dual_value = value
        else:
            self.parent._dual_value = value
