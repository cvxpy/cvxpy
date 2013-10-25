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
import operator as op
import numpy as np

class BinaryOperator(AffAtom):
    """
    Base class for expressions involving binary operators.
    """
    def __init__(self, lh_exp, rh_exp):
        super(BinaryOperator, self).__init__(lh_exp, rh_exp)

    def name(self):
        return ' '.join([self.args[0].name(), 
                         self.OP_NAME, 
                         self.args[1].name()])

    # Applies the binary operator to the values.
    def numeric(self, values):
        return self.OP_FUNC(values[0], values[1])
        
    # Sets the sign, curvature, and shape.
    def set_context(self):
        self._context = self.OP_FUNC(self.args[0]._context, self.args[1]._context)

    # Apply the binary operator to the arguments.
    @classmethod
    def graph_implementation(cls, var_args, size):
        obj = cls.OP_FUNC(var_args[0], var_args[1])
        return (obj, [])

    # Return the symbolic affine expression equal to the given index
    # into the expression.
    def index_object(self, key):
        # Scalar promotion
        promoted = self.promoted_index_object(key)
        if promoted is not None:
            return promoted
        return self.OP_FUNC(self.args[0][key], self.args[1][key])

    # Handle promoted scalars.
    def promoted_index_object(self, key):
        if self.args[0].size == (1,1):
            return self.OP_FUNC(self.args[0], self.args[1][key])
        elif self.args[1].size == (1,1):
            return self.OP_FUNC(self.args[0][key], self.args[1])
        else:
            return None

class AddExpression(BinaryOperator):
    OP_NAME = "+"
    OP_FUNC = op.add

class SubExpression(BinaryOperator):
    OP_NAME = "-"
    OP_FUNC = op.sub

class MulExpression(BinaryOperator):
    OP_NAME = "*"
    OP_FUNC = op.mul

    # Multiplies the values.
    def numeric(self, values):
        # Broadcast for promoted scalars.
        if max(values[0].shape) == 1 or max(values[1].shape) == 1:
            return values[0] * values[1]
        # Normal matrix multiplication.
        else:
            return np.dot(values[0], values[1])

    # Return the symbolic affine expression equal to the given index
    # in the expression.
    def index_object(self, key):
        # Scalar multiplication.
        promoted = self.promoted_index_object(key)
        if promoted is not None:
            return promoted
        # Matrix multiplication.
        return self.args[0][key[0],:]*self.args[1][:,key[1]]

    # If left-hand side is non-constant, replace lh*rh with x, x.T == rh.T*lh.T.
    def canonicalize(self):
        if not self.args[0].curvature.is_constant():
            x = Variable(*self.size)
            obj = x.canonical_form()[0]
            constraints = (x.T == self.args[1].T*self.args[0].T).canonical_form()[1]
            return (obj, constraints)
        else:
            return super(MulExpression, self).canonicalize()