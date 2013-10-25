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

from affine_atom import AffAtom
import operator as op

class UnaryOperator(AffAtom):
    """
    Base class for expressions involving unary operators. 
    """
    def __init__(self, expr):
        super(UnaryOperator, self).__init__(expr)

    def name(self):
        return self.OP_NAME + self.args[0].name()

    # Applies the unary operator to the value.
    def numeric(self, values):
        return self.OP_FUNC(values[0])
        
    # Sets the sign, curvature, and shape.
    def set_context(self):
        self._context = self.OP_FUNC(self.args[0]._context)

    # Apply the unary operator to the argument.
    @classmethod
    def graph_implementation(cls, var_args, size):
        return (cls.OP_FUNC(var_args[0]), [])

    # Apply the appropriate arithmetic operator to the expression
    # at the given index. Return the result.
    def index_object(self, key):
        return self.OP_FUNC(self.args[0][key])

class NegExpression(UnaryOperator):
    OP_NAME = "-"
    OP_FUNC = op.neg