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

from cvxpy.constraints.nonpos import NonPos
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class Zero(NonPos):
    """A constraint of the form :math:`x = 0`.

    The preferred way of creating a ``Zero`` constraint is through
    operator overloading. To constrain an expression ``x`` to be zero,
    simply write ``x == 0``. The former creates a ``Zero`` constraint with
    ``x`` as its argument.
    """
    def name(self):
        return "%s == 0" % self.args[0]

    def is_dcp(self):
        """A zero constraint is DCP if its argument is affine."""
        return self.args[0].is_affine()

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        Expression
        """
        if self.expr.value is None:
            return None
        return np.abs(self.expr.value)

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the EqConstraint.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self.args[0].canonical_form
        dual_holder = lu.create_eq(obj, constr_id=self.id)
        return (None, constraints + [dual_holder])
