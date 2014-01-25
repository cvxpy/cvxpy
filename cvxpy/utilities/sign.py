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

import bool_mat_utils as bu
import numpy as np

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
        POSITIVE_KEY: (np.bool_(False), np.bool_(True)),
        NEGATIVE_KEY: (np.bool_(True), np.bool_(False)),
        UNKNOWN_KEY: (np.bool_(True), np.bool_(True)),
        ZERO_KEY: (np.bool_(False), np.bool_(False)),
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

    def is_zero(self):
        """Is the expression all zero?
        """
        return self.is_positive() and self.is_negative()

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
        neg_mat = bu.dot(self.neg_mat, other.pos_mat) | \
                  bu.dot(self.pos_mat, other.neg_mat)
        pos_mat = bu.dot(self.neg_mat, other.neg_mat) | \
                  bu.dot(self.pos_mat, other.pos_mat)
        # Reduce 1x1 matrices to scalars.
        neg_mat = bu.to_scalar(neg_mat)
        pos_mat = bu.to_scalar(pos_mat)
        return Sign(neg_mat, pos_mat)

    def __neg__(self):
        """Equivalent to NEGATIVE * self.
        """
        return Sign(self.pos_mat, self.neg_mat)

    def __eq__(self, other):
        """Checks equality of arguments' attributes.
        """
        return np.all(self.neg_mat == other.neg_mat) and \
               np.all(self.pos_mat == other.pos_mat)

    def promote(self, rows, cols, keep_scalars=True):
        """Promotes the Sign's internal matrices to the desired size.

        Args:
            rows: The number of rows in the promoted internal matrices.
            cols: The number of columns in the promoted internal matrices.
            keep_scalars: Don't convert scalars to matrices.
        """
        neg_mat = bu.promote(self.neg_mat, rows, cols, keep_scalars)
        pos_mat = bu.promote(self.pos_mat, rows, cols, keep_scalars)
        return Sign(neg_mat, pos_mat)

    def __repr__(self):
        return "Sign(%s, %s)" % (self.neg_mat, self.pos_mat)

    def __str__(self):
        return "negative entries = %s, positive entries = %s" % \
            (self.neg_mat, self.pos_mat)

    def get_readable_repr(self, rows, cols):
        """Converts the internal representation to a matrix of strings.

        Args:
            rows: The number of rows in the expression.
            cols: The number of columns in the expression.

        Returns:
            A sign string or a Numpy 2D array of sign strings.
        """
        sign = self.promote(rows, cols, False)
        readable_mat = np.empty((rows, cols), dtype="object")
        for i in xrange(rows):
            for j in xrange(cols):
                # Is the entry unknown?
                if sign.pos_mat[i, j] and \
                     sign.neg_mat[i, j]:
                    readable_mat[i, j] = self.UNKNOWN_KEY
                # Is the entry positive?
                elif sign.pos_mat[i, j]:
                    readable_mat[i, j] = self.POSITIVE_KEY
                # Is the entry negative?
                elif sign.neg_mat[i, j]:
                    readable_mat[i, j] = self.NEGATIVE_KEY
                # The entry is zero.
                else:
                    readable_mat[i, j] = self.ZERO_KEY

        # Reduce readable_mat to a single string if homogeneous.
        if (readable_mat == readable_mat[0, 0]).all():
            return readable_mat[0, 0]
        else:
            return readable_mat

# Scalar signs.
Sign.POSITIVE = Sign.name_to_sign(Sign.POSITIVE_KEY)
Sign.NEGATIVE = Sign.name_to_sign(Sign.NEGATIVE_KEY)
Sign.UNKNOWN = Sign.name_to_sign(Sign.UNKNOWN_KEY)
Sign.ZERO = Sign.name_to_sign(Sign.ZERO_KEY)
