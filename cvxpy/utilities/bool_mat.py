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

class BoolMat(object):
    """ 
    A wrapper on a boolean numpy ndarray for use as a matrix
    to hold signs and curvatures.
    """
    # value - the underlying ndarray.
    def __init__(self, value):
        self.value = value

    # Cast an int to a bool, but pass any other type through.
    @staticmethod
    def cast_int(value):
        if isinstance(value, int):
            return value > 0
        else:
            return value

    # Return True if any entry is True.
    def any(self):
        return self.value.any()

    # For addition.
    def __or__(self, other):
        if isinstance(other, bool):
            if other: # Reduce to scalar
                return True
            else:
                return BoolMat(self.value)
        elif isinstance(other, BoolMat):
            return BoolMat(self.value | other.value)
        else:
            return NotImplemented

    # Handles boolean | BoolMat
    def __ror__(self, other):
        return self | other
        
    # For multiplication.
    def __mul__(self, other):
        if isinstance(other, bool):
            if other:
                return BoolMat(self.value)
            else: # Reduce to scalar.
                return False
        elif isinstance(other, BoolMat):
            mult_val = self.value.dot(other.value)
            return BoolMat(mult_val)
        else:
            return NotImplemented

    # Handles boolean * BoolMat
    def __rmul__(self, other):
        return self * other

    # For elementwise multiplication/bitwise and.
    def __and__(self, other):
        if isinstance(other, bool):
            return self * other
        elif isinstance(other, BoolMat):
            mult_val = self.value & other.value
            return BoolMat(mult_val)
        else:
            return NotImplemented

    # Handles boolean & BoolMat
    def __rand__(self, other):
        return self & other

    # For comparison.
    def __eq__(self, other):
        if isinstance(other, bool):
            return False
        elif isinstance(other, BoolMat):
            return self.value.shape == other.value.shape and \
                   (self.value == other.value).all()
        else:
            return NotImplemented

    # To string methods.
    def __repr__(self):
        return str(self.value)