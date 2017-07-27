"""
Copyright 2017 Steven Diamond

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

import cvxpy.utilities as u
from cvxpy.error import DCPError
from cvxpy.expressions.expression import Expression
import cvxpy.lin_ops.lin_utils as lu


class Minimize(u.Canonical):
    """An optimization objective for minimization.
    """

    NAME = "minimize"

    def __init__(self, expr):
        self.args = [Expression.cast_to_const(expr)]
        # Validate that the objective resolves to a scalar.
        if self.args[0].size != (1, 1):
            raise Exception("The '%s' objective must resolve to a scalar."
                            % self.NAME)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self.args[0]))

    def __str__(self):
        return ' '.join([self.NAME, self.args[0].name()])

    def __neg__(self):
        return Maximize(-self.args[0])

    def __add__(self, other):
        if not isinstance(other, (Minimize, Maximize)):
            return NotImplemented
        # Objectives must both be Minimize.
        if type(other) is Minimize:
            return Minimize(self.args[0] + other.args[0])
        else:
            raise DCPError("Problem does not follow DCP rules.")

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, (Minimize, Maximize)):
            return NotImplemented
        # Objectives must opposites
        return self + (-other)

    def __rsub__(self, other):
        if other == 0:
            return -self
        else:
            return NotImplemented

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        # If negative, reverse the direction of objective
        if (type(self) == Maximize) == (other < 0.0):
            return Minimize(self.args[0] * other)
        else:
            return Maximize(self.args[0] * other)

    __rmul__ = __mul__

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return self * (1.0/other)

    __truediv__ = __div__

    def canonicalize(self):
        """Pass on the target expression's objective and constraints.
        """
        return self.args[0].canonical_form

    def variables(self):
        """Returns the variables in the objective.
        """
        return self.args[0].variables()

    def parameters(self):
        """Returns the parameters in the objective.
        """
        return self.args[0].parameters()

    def constants(self):
        """Returns the constants in the objective.
        """
        return self.args[0].constants()

    def is_dcp(self):
        """The objective must be convex.
        """
        return self.args[0].is_convex()

    @property
    def value(self):
        """The value of the objective expression.
        """
        return self.args[0].value

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value.
        """
        return result


class Maximize(Minimize):
    """An optimization objective for maximization.
    """

    NAME = "maximize"

    def __neg__(self):
        return Minimize(-self.args[0])

    def __add__(self, other):
        if not isinstance(other, (Minimize, Maximize)):
            return NotImplemented
        # Objectives must both be Maximize.
        if type(other) is Maximize:
            return Maximize(self.args[0] + other.args[0])
        else:
            raise Exception("Problem does not follow DCP rules.")

    def canonicalize(self):
        """Negates the target expression's objective.
        """
        obj, constraints = super(Maximize, self).canonicalize()
        return (lu.neg_expr(obj), constraints)

    def is_dcp(self):
        """The objective must be concave.
        """
        return self.args[0].is_concave()

    def is_quadratic(self):
        """Returns if the objective is a quadratic function.
        """
        return self.args[0].is_quadratic()

    @staticmethod
    def primal_to_result(result):
        """The value of the objective given the solver primal value.
        """
        return -result
