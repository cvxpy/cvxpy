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

import expression

class BinaryOperator(object):
    """
    Base class for expressions involving binary operators.
    """
    def __init__(self, lh_exp, rh_exp):
        self.lh_exp = lh_exp
        self.rh_exp = expression.Expression.cast_to_const(rh_exp)
        super(BinaryOperator, self).__init__()

    def name(self):
        return ' '.join([self.lh_exp.name(), 
                         self.OP_NAME, 
                         self.rh_exp.name()])

class UnaryOperator(object):
    """
    Base class for expressions involving unary operators. 
    """
    def __init__(self, expr):
        self.expr = expr
        self._shape = expr._shape
        self._sign_curv = getattr(self.expr._sign_curv, self.OP_FUNC)()
        super(UnaryOperator, self).__init__()

    def name(self):
        return self.OP_NAME + self.expr.name()

    def canonicalize(self):
        obj,constraints = self.expr.canonical_form()
        obj = getattr(obj, self.OP_FUNC)()
        return (obj,constraints)

    @property
    def size(self):
        return self._shape.size

    # Apply the appropriate arithmetic operator to the expression
    # at the given index. Return the result.
    def index_object(self, key):
        return getattr(self.expr[key], self.OP_FUNC)()