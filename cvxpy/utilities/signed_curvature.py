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

class SignedCurvature(object):
    """ A data structure for the sign and curvature of an expression. """
    # sign - the signs of the expression's entries.
    # curvature - the curvatures of the expression's entries.
    def __init__(self, sign, curvature):
        self.sign = sign
        self.curvature = curvature

    """ Arithmetic operations """
    def __add__(self, other):
        sign = self.sign + other.sign
        curvature = self.curvature + other.curvature
        return SignedCurvature(sign, curvature)

    def __sub__(self, other):
        sign = self.sign - other.sign
        curvature = self.curvature - other.curvature
        return SignedCurvature(sign, curvature)

    # Assumes self has all constant curvature.
    def __mul__(self, other):
        sign = self.sign * other.sign
        curvature = other.curvature.sign_mul(sign)
        return SignedCurvature(sign, curvature)

    def __neg__(self):
        sign = -self.sign
        curvature = -self.curvature
        return SignedCurvature(sign, curvature)