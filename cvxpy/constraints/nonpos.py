"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.lin_ops.lin_utils as lu
# Only need Variable from expressions, but that would create a circular import.
from cvxpy.constraints.constraint import Constraint
import numpy as np


class NonPos(Constraint):
    """A constraint of the form :math:`x \leq 0`.

    The preferred way of creating a ``NonPos`` constraint is through
    operator overloading. To constrain an expression ``x`` to be non-positive,
    simply write ``x <= 0``; to constrain ``x`` to be non-negative, write
    ``x >= 0``. The former creates a ``NonPos`` constraint with ``x``
    as its argument, while the latter creates one with ``-x`` as its argument.
    Strict inequalities are not supported, as they do not make sense in a
    numerical setting.

    Parameters
    ----------
    expr : Expression
        The expression to constrain.
    constr_id : int
        A unique id for the constraint.
    """
    def __init__(self, expr, constr_id=None):
        if expr.is_complex():
            raise ValueError("Inequality constraints cannot be complex.")
        super(NonPos, self).__init__([expr], constr_id)

    def name(self):
        return "%s <= 0" % self.args[0]

    def is_dcp(self):
        """A non-positive constraint is DCP if its argument is convex."""
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

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        ---------
        NumPy.ndarray
        """
        if self.expr.value is None:
            return None
        return np.maximum(self.expr.value, 0)
