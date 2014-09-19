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
from cvxpy.expressions.expression import Expression
import cvxpy.lin_ops.lin_utils as lu

class Minimize(u.Canonical):
    """An optimization objective for minimization.
    """

    NAME = "minimize"

    def __init__(self, expr):
        self._expr = Expression.cast_to_const(expr)
        # Validate that the objective resolves to a scalar.
        if self._expr.size != (1, 1):
            raise Exception("The '%s' objective must resolve to a scalar."
                            % self.NAME)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._expr))

    def __str__(self):
        return ' '.join([self.NAME, self._expr.name()])

    def canonicalize(self):
        """Pass on the target expression's objective and constraints.
        """
        return self._expr.canonical_form

    def variables(self):
        """Returns the variables in the objective.
        """
        return self._expr.variables()

    def parameters(self):
        """Returns the parameters in the objective.
        """
        return self._expr.parameters()

    def is_dcp(self):
        """The objective must be convex.
        """
        return self._expr.is_convex()

    @property
    def value(self):
        """The value of the objective expression.
        """
        return self._expr.value

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value.
        """
        return result

class Maximize(Minimize):
    """An optimization objective for maximization.
    """

    NAME = "maximize"

    def canonicalize(self):
        """Negates the target expression's objective.
        """
        obj, constraints = super(Maximize, self).canonicalize()
        return (lu.neg_expr(obj), constraints)

    def is_dcp(self):
        """The objective must be concave.
        """
        return self._expr.is_concave()

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value.
        """
        return -result
