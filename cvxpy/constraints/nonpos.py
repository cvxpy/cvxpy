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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
# Only need Variable from expressions, but that would create a circular import.
from cvxpy.expressions import cvxtypes
from cvxpy.constraints.constraint import Constraint
import numpy as np


class NonPos(u.Canonical, Constraint):
    TOLERANCE = 1e-4

    def __init__(self, expr):
        self.dual_variable = cvxtypes.variable()(*expr.shape)
        super(NonPos, self).__init__([expr])

    @property
    def id(self):
        """Wrapper for compatibility with variables.
        """
        return self.constr_id

    def name(self):
        return "%s >= 0" % self.args[0]

    def __str__(self):
        """Returns a string showing the mathematical constraint.
        """
        return self.name()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s)" % (self.__class__.__name__,
                           repr(self.args[0]))

    def __nonzero__(self):
        """Raises an exception when called.

        Python 2 version.

        Called when evaluating the truth value of the constraint.
        Raising an error here prevents writing chained constraints.
        """
        return self._chain_constraints()

    def _chain_constraints(self):
        """Raises an error due to chained constraints.
        """
        raise Exception(
            ("Cannot evaluate the truth value of a constraint or "
             "chain constraints, e.g., 1 >= x >= 0.")
        )

    def __bool__(self):
        """Raises an exception when called.

        Python 3 version.

        Called when evaluating the truth value of the constraint.
        Raising an error here prevents writing chained constraints.
        """
        return self._chain_constraints()

    @property
    def shape(self):
        return self.args[0].shape

    @property
    def size(self):
        return np.prod(self.args[0].shape)

    # Left hand expression must be convex and right hand must be concave.
    def is_dcp(self):
        return self.args[0].is_convex()

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Marks the top level constraint as the dual_holder,
        so the dual value will be saved to the LeqConstraint.

        Returns
        -------
        tuple
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self.args[0].canonical_form
        dual_holder = lu.create_leq(obj, constr_id=self.id)
        return (None, constraints + [dual_holder])

    def variables(self):
        """Returns the variables in the compared expressions.
        """
        return self.args[0].variables()

    def parameters(self):
        """Returns the parameters in the compared expressions.
        """
        return self.args[0].parameters()

    def constants(self):
        """Returns the constants in the compared expressions.
        """
        return self.args[0].constants()

    @property
    def value(self):
        """Does the constraint hold?

        Returns
        -------
        bool
        """
        resid = self.residual.value
        if resid is None:
            return None
        else:
            return np.all(resid <= self.TOLERANCE)

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        Expression
        """
        return cvxtypes.neg()(self.args[0])

    @property
    def violation(self):
        """How much is this constraint off by?

        Returns
        -------
        NumPy matrix
        """
        return self.residual.value

    # The value of the dual variable.
    @property
    def dual_value(self):
        return self.dual_variable.value

    def save_value(self, value):
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        self.dual_variable.save_value(value)
