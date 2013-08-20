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

class SparseBoolMat(BoolMat):
    """ 
    A wrapper on a scipy sparse matrix to hold signs and curvatures.
    """
    # For addition.
    def __or__(self, other):
        if isinstance(other, bool):
            if other: # Reduce to scalar
                return True
            else:
                return SparseBoolMat(self.value)
        elif isinstance(other, (BoolMat, SparseBoolMat)):
            value = (self.value + other.value).astype('bool')
            return other.__class__(value) # Sparse + Dense = Dense
        else:
            return NotImplemented

    # For multiplication.
    def __and__(self, other):
        if isinstance(other, bool):
            if other:
                return SparseBoolMat(self.value)
            else: # Reduce to scalar.
                return False
        elif isinstance(other, (BoolMat,SparseBoolMat)):
            mult_val = self.value.dot(other.value).astype('bool')
            return other.__class__(mult_val) # Sparse * Dense = Dense
        else:
            return NotImplemented

    # Handles boolean & SparseBoolMat
    def __rand__(self, other):
        return self & other

    # For comparison.
    def __eq__(self, other):
        if isinstance(other, bool):
            return False
        elif isinstance(other, BoolMat):
            return self.todense() == other
        elif isinstance(other, SparseBoolMat):
            return self.value.shape == other.value.shape and \
                   (self.value == other.value).all()
        else:
            return NotImplemented

    # Utility function to convert a SparseBoolMat to a BoolMat.
    def todense(self):
        dense = self.value.astype('int64').todense() # Must be int64 for todense().
        dense = dense.astype('bool') # Convert back to bool.
        return BoolMat(dense)