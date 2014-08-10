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

class Sign(object):
    """Sign of convex optimization expressions.

    Attributes:
        sign_str: A string representation of the sign.
    """
    POSITIVE_KEY = 'POSITIVE'
    NEGATIVE_KEY = 'NEGATIVE'
    UNKNOWN_KEY = 'UNKNOWN'
    ZERO_KEY = 'ZERO'

    # List of valid sign strings.
    SIGN_STRINGS = [POSITIVE_KEY, NEGATIVE_KEY, UNKNOWN_KEY, ZERO_KEY]

    def __init__(self, sign_str):
        sign_str = sign_str.upper()
        if sign_str in Sign.SIGN_STRINGS:
            self.sign_str = sign_str
        else:
            raise ValueError("'%s' is not a valid sign name." % str(sign_str))

    @staticmethod
    def val_to_sign(val):
        """Converts a number to a sign.

        Args:
            val: A scalar.

        Returns:
            The Sign of val.
        """
        if val > 0:
            return Sign.POSITIVE
        elif val == 0:
            return Sign.ZERO
        else:
            return Sign.NEGATIVE

    def is_zero(self):
        """Is the expression all zero?
        """
        return self == Sign.ZERO

    def is_positive(self):
        """Is the expression positive?
        """
        return self.is_zero() or self == Sign.POSITIVE

    def is_negative(self):
        """Is the expression negative?
        """
        return self.is_zero() or self == Sign.NEGATIVE

    def is_unknown(self):
        """Is the expression sign unknown?
        """
        return self == Sign.UNKNOWN

    # Arithmetic operators
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
        if self.is_zero():
            return other
        elif self == Sign.POSITIVE and other.is_positive():
            return self
        elif self == Sign.NEGATIVE and other.is_negative():
            return self
        else:
            return Sign.UNKNOWN

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
        if self == Sign.ZERO or other == Sign.ZERO:
            return Sign.ZERO
        elif self == Sign.UNKNOWN or other == Sign.UNKNOWN:
            return Sign.UNKNOWN
        elif self != other:
            return Sign.NEGATIVE
        else:
            return Sign.POSITIVE

    def __neg__(self):
        """Equivalent to NEGATIVE * self.
        """
        return self * Sign.NEGATIVE

    def __eq__(self, other):
        """Checks equality of arguments' attributes.
        """
        return self.sign_str == other.sign_str

    def __ne__(self, other):
        """Checks equality of arguments' attributes.
        """
        return not (self.sign_str == other.sign_str)

    # To string methods.
    def __repr__(self):
        return "Sign('%s')" % self.sign_str

    def __str__(self):
        return self.sign_str

# Class constants for all sign types.
Sign.POSITIVE = Sign(Sign.POSITIVE_KEY)
Sign.NEGATIVE = Sign(Sign.NEGATIVE_KEY)
Sign.ZERO = Sign(Sign.ZERO_KEY)
Sign.UNKNOWN = Sign(Sign.UNKNOWN_KEY)
