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

from cvxpy.utilities.sparse_bool_mat import SparseBoolMat

class Sign(object):
    """Signs of the entries in an expression.

    Attributes:
        neg_mat: A boolean matrix indicating whether each entry is negative.
        pos_mat: A boolean matrix indicating whether each entry is positive.
    """

    POSITIVE_KEY = 'POSITIVE'
    NEGATIVE_KEY = 'NEGATIVE'
    UNKNOWN_KEY = 'UNKNOWN'
    ZERO_KEY = 'ZERO'

    # Map of sign string to scalar (neg_mat, pos_mat) values.
    # Zero is (False, False) and UNKNOWN is (True, True).
    SIGN_MAP = {
        POSITIVE_KEY: (SparseBoolMat.FALSE_MAT, SparseBoolMat.TRUE_MAT),
        NEGATIVE_KEY: (SparseBoolMat.TRUE_MAT, SparseBoolMat.FALSE_MAT),
        UNKNOWN_KEY: (SparseBoolMat.TRUE_MAT, SparseBoolMat.TRUE_MAT),
        ZERO_KEY: (SparseBoolMat.FALSE_MAT, SparseBoolMat.FALSE_MAT),
    }

    def __init__(self, neg_mat, pos_mat):
        self.neg_mat = neg_mat
        self.pos_mat = pos_mat

    @staticmethod
    def name_to_sign(sign_str):
        """Converts a sign name to a Sign object.

        Args:
            sign_str: A key in the SIGN_MAP.

        Returns:
            A Sign initialized with the selected value from SIGN_MAP.
        """
        sign_str = sign_str.upper()
        if sign_str in Sign.SIGN_MAP:
            return Sign(*Sign.SIGN_MAP[sign_str])
        else:
            raise Exception("'%s' is not a valid sign name." % str(sign_str))

    def is_positive(self):
        """Is the expression positive?
        """
        return not self.neg_mat.any()

    def is_negative(self):
        """Is the expression negative?
        """
        return not self.pos_mat.any()

    def __add__(self, other):
        """Handles the logic of adding signs.

        Cases:
            ZERO + ANYTHING = ANYTHING
            UNKNOWN + ANYTHING = UNKNOWN
            POSITIVE + NEGATIVE = UNKNOWN
            SAME + SAME = SAME

        Args:
            self: The Sign of the left-hand summand.
            other: The Sign of the right-hand summand.

        Returns:
            The Sign of the sum.
        """
        return Sign(
            self.neg_mat | other.neg_mat,
            self.pos_mat | other.pos_mat,
        )

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        """Handles logic of multiplying signs.

        Cases:
            ZERO * ANYTHING = ZERO
            UNKNOWN * NON-ZERO = UNKNOWN
            POSITIVE * NEGATIVE = NEGATIVE
            POSITIVE * POSITIVE = POSITIVE
            NEGATIVE * NEGATIVE = POSITIVE

        Args:
            self: The Sign of the left-hand multiplier.
            other: The Sign of the right-hand multiplier.

        Returns:
            The Sign of the product.
        """
        neg_mat = self.neg_mat * other.pos_mat | \
                  self.pos_mat * other.neg_mat
        pos_mat = self.neg_mat * other.neg_mat | \
                  self.pos_mat * other.pos_mat
        return Sign(neg_mat, pos_mat)

    def __neg__(self):
        """Equivalent to NEGATIVE * self.
        """
        return Sign(self.pos_mat, self.neg_mat)

    def __eq__(self, other):
        """Checks equality of arguments' attributes.
        """
        return self.neg_mat == other.neg_mat and self.pos_mat == other.pos_mat

    def promote(self, size):
        """Promotes the Sign's internal matrices to the desired size.
        """
        neg_mat = self.neg_mat.promote(size)
        pos_mat = self.pos_mat.promote(size)
        return Sign(neg_mat, pos_mat)

    def __repr__(self):
        return "Sign(%s, %s)" % (self.neg_mat, self.pos_mat)

    def __str__(self):
        return "negative entries = %s, positive entries = %s" % \
            (self.neg_mat, self.pos_mat)

# Scalar signs.
Sign.POSITIVE = Sign.name_to_sign(Sign.POSITIVE_KEY)
Sign.NEGATIVE = Sign.name_to_sign(Sign.NEGATIVE_KEY)
Sign.UNKNOWN = Sign.name_to_sign(Sign.UNKNOWN_KEY)
Sign.ZERO = Sign.name_to_sign(Sign.ZERO_KEY)
