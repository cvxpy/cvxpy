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
from cvxpy.constraints.constraint import Constraint
import numpy as np

class LeqConstraint(u.Canonical, Constraint):
    OP_NAME = "<="
    TOLERANCE = 1e-4
    def __init__(self, lh_exp, rh_exp):
        self.args = [lh_exp, rh_exp]
        self._expr = lh_exp - rh_exp
        self._dual_value = None
        super(LeqConstraint, self).__init__()

    @property
    def id(self):
        """Wrapper for compatibility with variables.
        """
        return self.constr_id

    def name(self):
        return ' '.join([str(self.args[0].name()),
                         self.OP_NAME,
                         str(self.args[1].name())])

    def __str__(self):
        """Returns a string showing the mathematical constraint.
        """
        return self.name()

    def __repr__(self):
        """Returns a string with information about the constraint.
        """
        return "%s(%s, %s)" % (self.__class__.__name__,
                               repr(self.args[0]),
                               repr(self.args[1]))

    def __nonzero__(self):
        """Raises an exception when called.

        Python 2 version.

        Called when evaluating the truth value of the constraint.
        Raising an error here prevents writing chained constraints.
        """
        raise Exception("Cannot evaluate the truth value of a constraint.")

    def __bool__(self):
        """Raises an exception when called.

        Python 3 version.

        Called when evaluating the truth value of the constraint.
        Raising an error here prevents writing chained constraints.
        """
        raise Exception("Cannot evaluate the truth value of a constraint.")

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

        Returns
        -------
        tuple
            A tuple of (affine expression, [constraints]).
        """
        obj, constraints = self._expr.canonical_form
        dual_holder = lu.create_leq(obj, constr_id=self.id)
        return (None, constraints + [dual_holder])

    def variables(self):
        """Returns the variables in the compared expressions.
        """
        return self._expr.variables()

    def parameters(self):
        """Returns the parameters in the compared expressions.
        """
        return self._expr.parameters()

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
            return np.all(self._expr.value <= self.TOLERANCE)

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
            return np.maximum(self._expr.value, 0)

    # The value of the dual variable.
    @property
    def dual_value(self):
        return self._dual_value

    def save_value(self, value):
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        self._dual_value = value
