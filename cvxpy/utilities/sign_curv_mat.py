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

class SignCurvMat(object):
    """ 
    A wrapper on a boolean numpy ndarray to mimic scalar behavior.
    """
    # value - the underlying ndarray.
    def __init__(self, value):
        self.value = value

    # For addition.
    def __or__(self, other):
        if isinstance(other, bool):
            return SignCurvMat(self.value | other)
        elif isinstance(other, SignCurvMat):
            return SignCurvMat(self.value | other.value)
        else:
            return NotImplemented

    # Handles boolean | SignCurvMat
    def __ror__(self, other):
        return self | other
        
    # For multiplication.
    def __and__(self, other):
        if isinstance(other, bool):
            return SignCurvMat(self.value & other)
        elif isinstance(other, SignCurvMat):
            mult_val = self.value.dot(other.value)
            # Reduce to scalar if possible.
            if mult_val.size == 1:
                mult_val = mult_val.item(0)
            return SignCurvMat(mult_val > 0)
        else:
            return NotImplemented

    # Handles boolean & SignCurvMat
    def __rand__(self, other):
        return self & other

    # For comparison.
    def __eq__(self, other):
        if isinstance(other, bool):
            return False
        elif isinstance(other, SignCurvMat):
            return self.value.shape == other.value.shape and \
                   (self.value == other.value).all()
        else:
            return NotImplemented