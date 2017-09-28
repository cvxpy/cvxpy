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

from cvxpy.constraints.constraint import Constraint
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class Zero(Constraint):
    """A constraint of the form :math:`x = 0`.

    The preferred way of creating a ``Zero`` constraint is through
    operator overloading. To constrain an expression ``x`` to be zero,
    simply write ``x == 0``. The former creates a ``Zero`` constraint with
    ``x`` as its argument.
    """
    def __init__(self, expr, constr_id=None):
        super(Zero, self).__init__([expr], constr_id)

    def __str__(self):
        """Returns a string showing the mathematical constraint.
        """
        return self.name()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s)" % (self.__class__.__name__,
                           repr(self.args[0]))

    @property
    def shape(self):
        """int : The shape of the constrained expression."""
        return self.args[0].shape

    @property
    def size(self):
        """int : The size of the constrained expression."""
        return self.args[0].size

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

    # The value of the dual variable.
    @property
    def dual_value(self):
        """NumPy.ndarray : The value of the dual variable.
        """
        return self.dual_variables[0].value

    # TODO(akshayka): Rename to save_dual_value to avoid collision with
    # value as defined above.
    def save_value(self, value):
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        self.dual_variables[0].save_value(value)
