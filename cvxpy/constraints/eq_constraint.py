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

from cvxpy.constraints.leq_constraint import LeqConstraint
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class EqConstraint(LeqConstraint):
    OP_NAME = "=="
    # Both sides must be affine.
    def is_dcp(self):
        return self._expr.is_affine()

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
            return np.all(np.abs(self._expr.value) <= self.TOLERANCE)

    @property
    def violation(self):
        """How much is this constraint off by?

        Returns
        -------
        NumPy matrix
        """
        if self._expr.value is None:
            return None
        else:
            return np.abs(self._expr.value)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the EqConstraint.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self._expr.canonical_form
        dual_holder = lu.create_eq(obj, constr_id=self.id)
        return (None, constraints + [dual_holder])
