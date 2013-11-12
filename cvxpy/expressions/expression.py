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

from .. import settings as s
from .. import utilities as u
from .. import interface as intf
from ..constraints import EqConstraint, LeqConstraint
import types
import abc

# Casts the second argument of a binary operator as an Expression.
def cast_other(binary_op):
    def cast_op(self, other):
        other = self.cast_to_const(other)
        return binary_op(self, other)
    return cast_op

class Expression(object):
    """
    A mathematical expression in a convex optimization problem.
    """
    __metaclass__ = abc.ABCMeta
    # Returns the graph implementation of the expression:
    # a tuple of (AffExpression, [constraints]).
    @abc.abstractmethod
    def canonicalize(self):
        return NotImplemented

    # Returns the value of the expression.
    # Simulates recursion with stack to avoid stack overflow.
    @property
    def value(self):
        stack = [self.starting_state()]
        while True:
            node = stack[-1]["expression"]
            if stack[-1]["index"] >= len(node.subexpressions):
                value = node.numeric(stack[-1]["values"])
                # Break if the node does not have a value.
                if value is None:
                    return None
                stack.pop()
                if len(stack) > 0:
                    stack[-1]["values"].append(value)
                else:
                    return value
            else:
                next = node.subexpressions[stack[-1]["index"]]
                stack[-1]["index"] += 1
                stack.append(next.starting_state())

    # Helper function for value.
    # Returns the starting state dict for the expression.
    def starting_state(self):
        return {"expression": self,
                "index": 0,
                "values": [],
        }

    # Applies the argument for the expression to the values.
    @abc.abstractmethod
    def numeric(self, values):
        return NotImplemented

    # TODO priority
    def __repr__(self):
        return self.name()

    # Returns string representation of the expression.
    @abc.abstractmethod
    def name(self):
        return NotImplemented

    # Returns the DCP attributes of the expression,
    # i.e. curvature, sign, and shape.
    @abc.abstractmethod
    def _dcp_attr(self):
        return NotImplemented

    # The curvature of the expression.
    @property
    def curvature(self):
        return self._dcp_attr().curvature

    # The sign of the expression, a (row,col) tuple.
    @property
    def sign(self):
        return self._dcp_attr().sign

    # The shape of the expression, an object.
    @property
    def shape(self):
        return self._dcp_attr().shape

    # The dimensions of the expression.
    @property
    def size(self):
        return self.shape.size

    # Is the expression a scalar?
    def is_scalar(self):
        return self.size == (1,1)

    # Is the expression a column vector?
    def is_vector(self):
        return self.size[1] == 1

    # Return a slice/index into the expression.
    def __getitem__(self, key):
        # Indexing into a scalar returns the scalar.
        if self.size == (1,1):
            return self
        else:
            return types.index()(self, key)

    # Iterating over the leaf returns the index expressions
    # in column major order.
    def __iter__(self):
        for col in range(self.size[1]):
            for row in range(self.size[0]):
                yield self[row,col]

    def __len__(self):
        length = self.size[0]*self.size[1]
        if length == 1: # Numpy will iterate over anything with a length.
            return NotImplemented
        else:
            return length

    # The transpose of the expression.
    @property
    def T(self):
        if self.size == (1,1): # Transpose of a scalar is that scalar.
            return self
        else:
            return types.transpose()(self)

    """ Arithmetic operators """
    # Cast to Constant if not an Expression.
    @staticmethod
    def cast_to_const(expr):
        return expr if isinstance(expr, Expression) else types.constant()(expr)

    @cast_other
    def __add__(self, other):
        return types.add_expr()(self, other)

    # Called for Number + Expression.
    @cast_other
    def __radd__(self, other):
        return types.add_expr()(other, self)

    @cast_other
    def __sub__(self, other):
        return types.sub_expr()(self, other)

    # Called for Number - Expression.
    @cast_other
    def __rsub__(self, other):
        return types.sub_expr()(other, self)

    @cast_other
    def __mul__(self, other):
        return types.mul_expr()(self, other)

    # Called for Number * Expression.
    @cast_other
    def __rmul__(self, other):
        return types.mul_expr()(other, self)

    def __neg__(self):
        return types.neg_expr()(self)

    """ Comparison operators """
    @cast_other
    def __eq__(self, other):
        return EqConstraint(self, other)

    @cast_other
    def __le__(self, other):
        return LeqConstraint(self, other)

    @cast_other
    def __ge__(self, other):
        return other.__le__(self)