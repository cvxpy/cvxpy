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

class Curvature(object):
    """Curvature for a convex optimization expression.

    Attributes:
        curvature_str: A string representation of the curvature.
    """
    CONSTANT_KEY = 'CONSTANT'
    AFFINE_KEY = 'AFFINE'
    CONVEX_KEY = 'CONVEX'
    CONCAVE_KEY = 'CONCAVE'
    UNKNOWN_KEY = 'UNKNOWN'

    # List of valid curvature strings.
    CURVATURE_STRINGS = [CONSTANT_KEY, AFFINE_KEY, CONVEX_KEY,
                         CONCAVE_KEY, UNKNOWN_KEY]
    # For multiplying curvature by negative sign.
    NEGATION_MAP = {CONVEX_KEY: CONCAVE_KEY, CONCAVE_KEY: CONVEX_KEY}

    def __init__(self, curvature_str):
        """Converts a curvature name to a Curvature object.

        Args:
            curvature_str: A key in the CURVATURE_MAP.

        Returns:
            A Curvature initialized with the selected value from CURVATURE_MAP.
        """
        curvature_str = curvature_str.upper()
        if curvature_str in Curvature.CURVATURE_STRINGS:
            self.curvature_str = curvature_str
        else:
            raise Error("'%s' is not a valid curvature name." %
                        str(curvature_str))

    def __repr__(self):
        return "Curvature('%s')" % self.curvature_str

    def __str__(self):
        return self.curvature_str

    def is_constant(self):
        """Is the expression constant?
        """
        return self == Curvature.CONSTANT

    def is_affine(self):
        """Is the expression affine?
        """
        return self.is_constant() or self == Curvature.AFFINE

    def is_convex(self):
        """Is the expression convex?
        """
        return self.is_affine() or self == Curvature.CONVEX

    def is_concave(self):
        """Is the expression concave?
        """
        return self.is_affine() or self == Curvature.CONCAVE

    def is_unknown(self):
        """Is the expression unknown curvature?
        """
        return self == Curvature.UNKNOWN

    def is_dcp(self):
        """Is the expression DCP compliant? (i.e., no unknown curvatures).
        """
        return not self.is_unknown()

    def __add__(self, other):
        """Handles the logic of adding curvatures.

        Cases:
          CONSTANT + ANYTHING = ANYTHING
          AFFINE + NONCONSTANT = NONCONSTANT
          CONVEX + CONCAVE = UNKNOWN
          SAME + SAME = SAME

        Args:
            self: The Curvature of the left-hand summand.
            other: The Curvature of the right-hand summand.

        Returns:
            The Curvature of the sum.
        """
        if self.is_constant():
            return other
        elif self.is_affine() and other.is_affine():
            return Curvature.AFFINE
        elif self.is_convex() and other.is_convex():
            return Curvature.CONVEX
        elif self.is_concave() and other.is_concave():
            return Curvature.CONCAVE
        else:
            return Curvature.UNKNOWN

    def __sub__(self, other):
        return self + -other

    @staticmethod
    def sign_mul(sign, curv):
        """Handles logic of sign by curvature multiplication.

        Cases:
            ZERO * ANYTHING = CONSTANT
            NON-ZERO * AFFINE/CONSTANT = AFFINE/CONSTANT
            UNKNOWN * NON-AFFINE = UNKNOWN
            POSITIVE * ANYTHING = ANYTHING
            NEGATIVE * CONVEX = CONCAVE
            NEGATIVE * CONCAVE = CONVEX

        Args:
            sign: The Sign of the left-hand multiplier.
            curv: The Curvature of the right-hand multiplier.

        Returns:
            The Curvature of the product.
        """
        if sign.is_zero():
            return Curvature.CONSTANT
        elif sign.is_positive() or curv.is_affine():
            return curv
        elif sign.is_negative():
            curvature_str = Curvature.NEGATION_MAP.get(curv.curvature_str,
                                                       curv.curvature_str)
            return Curvature(curvature_str)
        else: # sign is unknown.
            return Curvature.UNKNOWN

    def __neg__(self):
        """Equivalent to NEGATIVE * self.
        """
        curvature_str = Curvature.NEGATION_MAP.get(self.curvature_str,
                                                   self.curvature_str)
        return Curvature(curvature_str)

    def __eq__(self, other):
        """Are the two curvatures equal?
        """
        return self.curvature_str == other.curvature_str

    def __ne__(self, other):
        """Are the two curvatures not equal?
        """
        return self.curvature_str != other.curvature_str

# Class constants for all curvature types.
Curvature.CONSTANT = Curvature(Curvature.CONSTANT_KEY)
Curvature.AFFINE = Curvature(Curvature.AFFINE_KEY)
Curvature.CONVEX = Curvature(Curvature.CONVEX_KEY)
Curvature.CONCAVE = Curvature(Curvature.CONCAVE_KEY)
Curvature.UNKNOWN = Curvature(Curvature.UNKNOWN_KEY)
Curvature.NONCONVEX = Curvature(Curvature.UNKNOWN_KEY)
