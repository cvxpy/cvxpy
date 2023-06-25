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

import numpy as np

from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities import scopes


class Zero(Constraint):
    """A constraint of the form :math:`x = 0`.

    The preferred way of creating a ``Zero`` constraint is through
    operator overloading. To constrain an expression ``x`` to be zero,
    simply write ``x == 0``. The former creates a ``Zero`` constraint with
    ``x`` as its argument.
    """
    def __init__(self, expr, constr_id=None) -> None:
        super(Zero, self).__init__([expr], constr_id)

    def __str__(self):
        """Returns a string showing the mathematical constraint.
        """
        return self.name()

    def __repr__(self) -> str:
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

    def name(self) -> str:
        return "%s == 0" % self.args[0]

    def is_dcp(self, dpp: bool = False) -> bool:
        """A zero constraint is DCP if its argument is affine."""
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine()
        return self.args[0].is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

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

    # The value of the dual variable.
    @property
    def dual_value(self):
        """NumPy.ndarray : The value of the dual variable.
        """
        return self.dual_variables[0].value

    def save_dual_value(self, value) -> None:
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        self.dual_variables[0].save_value(value)


class Equality(Constraint):
    """A constraint of the form :math:`x = y`.
    """
    def __init__(self, lhs, rhs, constr_id=None) -> None:
        self._expr = lhs - rhs
        super(Equality, self).__init__([lhs, rhs], constr_id)

    def __str__(self):
        """Returns a string showing the mathematical constraint.
        """
        return self.name()

    def __repr__(self) -> str:
        """Returns a string with information about the constraint.
        """
        return "%s(%s, %s)" % (self.__class__.__name__,
                               repr(self.args[0]), repr(self.args[1]))

    def _construct_dual_variables(self, args) -> None:
        super(Equality, self)._construct_dual_variables([self._expr])

    @property
    def expr(self):
        return self._expr

    @property
    def shape(self):
        """int : The shape of the constrained expression."""
        return self.expr.shape

    @property
    def size(self):
        """int : The size of the constrained expression."""
        return self.expr.size

    def name(self) -> str:
        return "%s == %s" % (self.args[0], self.args[1])

    def is_dcp(self, dpp: bool = False) -> bool:
        """An equality constraint is DCP if its argument is affine."""
        if dpp:
            with scopes.dpp_scope():
                return self.expr.is_affine()
        return self.expr.is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                return (self.args[0].is_log_log_affine() and
                        self.args[1].is_log_log_affine())
        return (self.args[0].is_log_log_affine() and
                self.args[1].is_log_log_affine())

    def is_dqcp(self) -> bool:
        return self.is_dcp()

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

    @property
    def dual_value(self):
        """NumPy.ndarray : The value of the dual variable.
        """
        return self.dual_variables[0].value

    def save_dual_value(self, value) -> None:
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        self.dual_variables[0].save_value(value)
