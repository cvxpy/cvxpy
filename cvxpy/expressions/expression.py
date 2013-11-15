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

from .. import interface as intf
from .. import utilities as u
from ..utilities import performance_utils as pu
from ..constraints import EqConstraint, LeqConstraint
import types
import abc

def _cast_other(binary_op):
    """Casts the second argument of a binary operator as an Expression.

    Args:
        binary_op: A binary operator in the Expression class.

    Returns:
        A wrapped binary operator that can handle non-Expression arguments.
    """
    def cast_op(self, other):
        """A wrapped binary operator that can handle non-Expression arguments.
        """
        other = self.cast_to_const(other)
        return binary_op(self, other)
    return cast_op

class Expression(u.Canonical):
    """
    A mathematical expression in a convex optimization problem.
    """

    __metaclass__ = abc.ABCMeta

    @property
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A cvxopt sparse matrix or a scalar.
        """
        # Simulates recursion with stack to avoid stack overflow.
        # TODO switch to recursive.
        stack = [self._starting_state()]
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
                    # Reduce 1x1 matrices to scalars.
                    if intf.size(value) == (1,1):
                        return intf.scalar_value(value)
                    else:
                        return value
            else:
                next_node = node.subexpressions[stack[-1]["index"]]
                stack[-1]["index"] += 1
                stack.append(next_node._starting_state())

    def _starting_state(self):
        """Helper function for value.

        Returns:
            The starting state dict for the expression.
        """
        return {"expression": self,
                "index": 0,
                "values": [],
        }

    def __repr__(self):
        """TODO priority
        """
        return self.name()

    @abc.abstractmethod
    def name(self):
        """Returns the string representation of the expression.
        """
        return NotImplemented

    @property
    def curvature(self):
        """ Returns the curvature of the expression.
        """
        return self._dcp_attr.curvature

    @property
    def sign(self):
        """ Returns the sign of the expression.
        """
        return self._dcp_attr.sign

    # The shape of the expression, an object.
    @property
    def shape(self):
        """ Returns the shape of the expression.
        """
        return self._dcp_attr.shape

    @property
    def size(self):
        """ Returns the (row, col) dimensions of the expression.
        """
        return self.shape.size

    def is_scalar(self):
        """Is the expression a scalar?
        """
        return self.size == (1, 1)

    def is_vector(self):
        """Is the expression a column vector?
        """
        return self.size[1] == 1

    def __getitem__(self, key):
        """Return a slice/index into the expression.
        """
        # Indexing into a scalar returns the scalar.
        if self.size == (1, 1):
            return self
        else:
            return types.index()(self, key)

    def __iter__(self):
        """Yields indices into the expression in column major order.
        """
        for col in range(self.size[1]):
            for row in range(self.size[0]):
                yield self[row, col]

    def __len__(self):
        """The number of entries in a matrix expression.
        """
        length = self.size[0]*self.size[1]
        if length == 1: # Numpy will iterate over anything with a length.
            return NotImplemented
        else:
            return length

    @property
    def T(self):
        """The transpose of an expression.
        """
        # Transpose of a scalar is that scalar.
        if self.size == (1, 1):
            return self
        else:
            return types.transpose()(self)

    # Arithmetic operators.
    @staticmethod
    def cast_to_const(expr):
        """Converts a non-Expression to a Constant.
        """
        return expr if isinstance(expr, Expression) else types.constant()(expr)

    @_cast_other
    def __add__(self, other):
        """The sum of two expressions.
        """
        return types.add_expr()(self, other)

    @_cast_other
    def __radd__(self, other):
        """Called for Number + Expression.
        """
        return other + self

    @_cast_other
    def __sub__(self, other):
        """The difference of two expressions.
        """
        return self + -other

    @_cast_other
    def __rsub__(self, other):
        """Called for Number - Expression.
        """
        return other - self

    @_cast_other
    def __mul__(self, other):
        """The product of two expressions.
        """
        # Cannot multiply two non-constant expressions.
        if not self.curvature.is_constant() and \
           not other.curvature.is_constant():
            # TODO replace with special exception.
            raise Exception("Cannot multiply two non-constants.")
        # The constant term must always be on the left.
        elif not self.curvature.is_constant():
            return (other.T * self.T).T
        else:
            return types.mul_expr()(self, other)

    @_cast_other
    def __rmul__(self, other):
        """Called for Number * Expression.
        """
        return other * self

    def __neg__(self):
        """The negation of the expression.
        """
        return types.neg_expr()(self)

    # Comparison operators.
    @_cast_other
    def __eq__(self, other):
        """Returns an equality constraint.
        """
        return EqConstraint(self, other)

    @_cast_other
    def __le__(self, other):
        """Returns an inequality constraint.
        """
        return LeqConstraint(self, other)

    @_cast_other
    def __ge__(self, other):
        """Returns an inequality constraint.
        """
        return other.__le__(self)
