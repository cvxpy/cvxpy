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
    # signs - the signs of the expression's entries.
    # curvatures - the curvatures of the expression's entries.
    def __init__(self, signs, curvatures):
        self.signs = signs
        self.curvatures = curvatures

    """ Arithmetic operations """
    def __add__(self, other):
        signs = self.signs + other.signs
        curvatures = self.curvatures + other.curvatures
        return SignedCurvature(signs, curvatures)

    def __sub__(self, other):
        signs = self.signs - other.signs
        curvatures = self.curvatures - other.curvatures
        return SignedCurvature(signs, curvatures)

    # Assumes self has all constant curvature.
    def __mul__(self, other):
        signs = self.signs * other.signs
        curvatures = self.signs * other.curvatures
        return SignedCurvature(signs, curvatures)

    def __neg__(self):
        signs = -self.signs
        curvatures = -self.curvatures
        return SignedCurvature(signs, curvatures)

    # # Helper to prevent promotion to a matrix when possible.
    # @staticmethod
    # def prevent_promotion(lh, rh, types_to_match):
    #     if len(lh) == 1 and lh[0] in types_to_match:
    #         return lh
    #     elif len(rh) == 1 and rh[0] in types_to_match:
    #         return rh
    #     else:
    #         return None