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

from sign import Sign
from curvature import Curvature

class Context(object):
    """ A data structure for the sign, curvature, and shape of an expression. """
    # sign - the signs of the expression's entries.
    # curvature - the curvatures of the expression's entries.
    # shape - the dimensions of the expression.
    def __init__(self, sign, curvature, shape):
        self.sign = sign
        self.curvature = curvature
        self.shape = shape

    """ Arithmetic operations """
    def __add__(self, other):
        shape = self.shape + other.shape
        sign = self.sign + other.sign
        curvature = self.curvature + other.curvature
        return Context(sign, curvature, shape)

    def __sub__(self, other):
        shape = self.shape + other.shape
        sign = self.sign - other.sign
        curvature = self.curvature - other.curvature
        return Context(sign, curvature, shape)

    # Assumes self has all constant curvature.
    def __mul__(self, other):
        shape = self.shape * other.shape
        sign = Sign.mul(self.sign, self.shape.size,
                        other.sign, other.shape.size)
        curvature = Curvature.sign_mul(self.sign, self.shape.size,
                                       other.curvature, other.shape.size)
        return Context(sign, curvature, shape)

    def __neg__(self):
        return Context(-self.sign, -self.curvature, self.shape)