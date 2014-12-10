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

from cvxpy.utilities.curvature import Curvature
from cvxpy.utilities.shape import Shape
from cvxpy.utilities.sign import Sign

class DCPAttr(object):
    """ A data structure for the sign, curvature, and shape of an expression.

    Attributes:
        sign: The signs of the entries in the matrix expression.
        curvature: The curvatures of the entries in the matrix expression.
        shape: The dimensions of the matrix expression.
    """

    def __init__(self, sign, curvature, shape):
        self.sign = sign
        self.curvature = curvature
        self.shape = shape

    def __add__(self, other):
        """Determines the DCP attributes of two expressions added together.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the sum.
        """
        shape = self.shape + other.shape
        sign = self.sign + other.sign
        curvature = self.curvature + other.curvature
        return DCPAttr(sign, curvature, shape)

    def __sub__(self, other):
        """Determines the DCP attributes of one expression minus another.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the difference.
        """
        shape = self.shape + other.shape
        sign = self.sign - other.sign
        curvature = self.curvature - other.curvature
        return DCPAttr(sign, curvature, shape)

    def __mul__(self, other):
        """Determines the DCP attributes of two expressions multiplied together.

        Assumes one of the arguments has constant curvature.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the product.
        """
        shape = self.shape * other.shape
        sign = self.sign * other.sign
        if self.curvature.is_constant():
            curvature = Curvature.sign_mul(self.sign, other.curvature)
        else:
            curvature = Curvature.sign_mul(other.sign, self.curvature)
        return DCPAttr(sign, curvature, shape)

    @staticmethod
    def mul_elemwise(lh_expr, rh_expr):
        """Determines the DCP attributes of expressions multiplied elementwise.

        Assumes the left-hand argument has constant curvature and both
        arguments have the same shape.

        Args:
            lh_expr: The DCPAttr of the left-hand expression.
            rh_expr: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the product.
        """
        shape = lh_expr.shape + rh_expr.shape
        sign = lh_expr.sign * rh_expr.sign
        curvature = Curvature.sign_mul(lh_expr.sign, rh_expr.curvature)
        return DCPAttr(sign, curvature, shape)

    def __div__(self, other):
        """Determines the DCP attributes of one expression divided by another.

        Assumes the right-hand argument has constant curvature.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the product.
        """
        return other*self

    def __truediv__(self, other):
        """Determines the DCP attributes of one expression divided by another.

        Assumes the right-hand argument has constant curvature.

        Args:
            self: The DCPAttr of the left-hand expression.
            other: The DCPAttr of the right-hand expression.

        Returns:
            The DCPAttr of the product.
        """
        return other*self

    def __neg__(self):
        """Determines the DCP attributes of a negated expression.
        """
        return DCPAttr(-self.sign, -self.curvature, self.shape)
