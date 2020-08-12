"""
Copyright, the CVXPY authors

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

# Only need Variable from expressions, but that would create a circular import.
from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities import scopes
import numpy as np


class NonPos(Constraint):
    """A constraint of the form :math:`x \\leq 0`.

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
        super(NonPos, self).__init__([expr], constr_id)

    def name(self):
        return "%s <= 0" % self.args[0]

    def is_dcp(self, dpp=False):
        """A non-positive constraint is DCP if its argument is convex."""
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_convex()
        return self.args[0].is_convex()

    def is_dgp(self, dpp=False):
        return False

    def is_dqcp(self):
        return self.args[0].is_quasiconvex()

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


class Inequality(Constraint):
    """A constraint of the form :math:`x \\leq y`.

    Parameters
    ----------
    expr : Expression
        The expression to constrain.
    constr_id : int
        A unique id for the constraint.
    """
    def __init__(self, lhs, rhs, constr_id=None):
        self._expr = lhs - rhs
        # TODO remove this restriction.
        if self._expr.is_complex():
            raise ValueError("Inequality constraints cannot be complex.")
        super(Inequality, self).__init__([lhs, rhs], constr_id)

    def _construct_dual_variables(self, args):
        super(Inequality, self)._construct_dual_variables([self._expr])

    @property
    def expr(self):
        return self._expr

    def name(self):
        return "%s <= %s" % (self.args[0], self.args[1])

    @property
    def shape(self):
        """int : The shape of the constrained expression."""
        return self.expr.shape

    @property
    def size(self):
        """int : The size of the constrained expression."""
        return self.expr.size

    def is_dcp(self, dpp=False):
        """A non-positive constraint is DCP if its argument is convex."""
        if dpp:
            with scopes.dpp_scope():
                return self.expr.is_convex()
        return self.expr.is_convex()

    def is_dgp(self, dpp=False):
        if dpp:
            with scopes.dpp_scope():
                return (self.args[0].is_log_log_convex() and
                        self.args[1].is_log_log_concave())
        return (self.args[0].is_log_log_convex() and
                self.args[1].is_log_log_concave())

    def is_dpp(self, context='dcp'):
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError('Unsupported context ', context)

    def is_dqcp(self):
        return (
            self.is_dcp() or
            (self.args[0].is_quasiconvex() and self.args[1].is_constant()) or
            (self.args[0].is_constant() and self.args[1].is_quasiconcave()))

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
