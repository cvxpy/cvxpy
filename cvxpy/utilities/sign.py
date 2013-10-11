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

from bool_mat import BoolMat
import bool_mat_utils as bu

class Sign(object):
    """ 
    Signs of the entries in an expression.
    """
    POSITIVE_KEY = 'POSITIVE'
    NEGATIVE_KEY = 'NEGATIVE'
    UNKNOWN_KEY = 'UNKNOWN'
    ZERO_KEY = 'ZERO'

    # Map of sign string to scalar (neg_mat, pos_mat) values.
    # Zero is (False, False) and UNKNOWN is (True, True).
    SIGN_MAP = {
        POSITIVE_KEY: (False, True),
        NEGATIVE_KEY: (True, False),
        UNKNOWN_KEY: (True, True),
        ZERO_KEY: (False, False),
    }

    # neg_mat - a boolean matrix indicating whether each entry is negative.
    # pos_mat - a boolean matrix indicating whether each entry is positive.
    def __init__(self, neg_mat, pos_mat):
        self.neg_mat = neg_mat
        self.pos_mat = pos_mat

    @staticmethod
    def name_to_sign(sign_str):
        sign_str = sign_str.upper()
        if sign_str in Sign.SIGN_MAP:
            return Sign(*Sign.SIGN_MAP[sign_str])
        else:
            raise Exception("'%s' is not a valid sign name." % str(sign_str))

    # Is the expression positive?
    def is_positive(self):
        return not bu.any(self.neg_mat)

    # Is the expression negative?
    def is_negative(self):
        return not bu.any(self.pos_mat)

    # Arithmetic operators.
    """
    Handles logic of sign addition:
        ZERO + ANYTHING = ANYTHING
        UNKNOWN + ANYTHING = UNKNOWN
        POSITIVE + NEGATIVE = UNKNOWN
        SAME + SAME = SAME
    """
    def __add__(self, other):
        return Sign(
            self.neg_mat | other.neg_mat,
            self.pos_mat | other.pos_mat,
        )
    
    def __sub__(self, other):
        return self + -other
    
    """
    Handles logic of sign multiplication:
        ZERO * ANYTHING = ZERO
        UNKNOWN * NON-ZERO = UNKNOWN
        POSITIVE * NEGATIVE = NEGATIVE
        POSITIVE * POSITIVE = POSITIVE
        NEGATIVE * NEGATIVE = POSITIVE
    """
    @staticmethod
    def mul(lh_sign, lh_size, rh_sign, rh_size):
        neg_mat = bu.mul(lh_sign.neg_mat, lh_size, 
                              rh_sign.pos_mat, rh_size) | \
                  bu.mul(lh_sign.pos_mat, lh_size, 
                              rh_sign.neg_mat, rh_size)
        pos_mat = bu.mul(lh_sign.neg_mat, lh_size,
                              rh_sign.neg_mat, rh_size) | \
                  bu.mul(lh_sign.pos_mat, lh_size,
                              rh_sign.pos_mat, rh_size)
        return Sign(neg_mat, pos_mat)
    
    # Equivalent to NEGATIVE * self
    def __neg__(self):
        return Sign(self.pos_mat, self.neg_mat)

    # Comparison.
    def __eq__(self, other):
        return self.neg_mat == other.neg_mat and self.pos_mat == other.pos_mat

    # Promotes neg_mat and pos_mat to BoolMats of the given size.
    def promote(self, size):
        neg_mat = BoolMat.promote(self.neg_mat, size)
        pos_mat = BoolMat.promote(self.pos_mat, size)
        return Sign(neg_mat, pos_mat)
        
    # To string methods.
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