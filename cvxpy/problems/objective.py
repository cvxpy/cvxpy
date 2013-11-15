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
from .. import interface as intf
from ..expressions.expression import Expression

class Minimize(object):
    """
    An optimization objective for minimization.
    """
    NAME = "minimize"

    # expr - the expression to minimize.
    def __init__(self, expr):
        self.expr = Expression.cast_to_const(expr)
        # Validate that the objective resolves to a scalar.
        if self.expr.size != (1,1):
            raise Exception("The objective '%s' must resolve to a scalar." 
                            % self)

    def __repr__(self):
        return self.name()

    def name(self):
        return ' '.join([self.NAME, self.expr.name()])

    # Pass on the target expression's objective and constraints.
    def canonicalize(self):
        return self.expr.canonicalize()

    # Objective must be convex.
    def is_dcp(self):
        return self.expr.curvature.is_convex()

    @property
    def value(self):
        """The value of the objective expression.
        """
        return self.expr.value

    # The value of the objective given the solver primal value.
    def _primal_to_result(self, result):
        return result

class Maximize(Minimize):
    NAME = "maximize"
    """
    An optimization objective for maximization.
    """
    def canonicalize(self):
        obj,constraints = super(Maximize, self).canonicalize()
        return (-obj, constraints)

    # Objective must be concave.
    def is_dcp(self):
        return self.expr.curvature.is_concave()

    # The value of the objective given the solver primal value.
    def _primal_to_result(self, result):
        return -result