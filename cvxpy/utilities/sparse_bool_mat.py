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

from scipy import sparse
import numpy as np

def reduce_result(func):
    """
    Wraps a SparseBoolMat function so that the result
    is reduced to a 1x1 matrix if possible.
    """
    def wrapped(*args):
        """
        A wrapper on the SparseBoolMat function that
        reduces the result to a 1x1 matrix if possible.
        """
        result = func(*args)
        return result.reduce()
    return wrapped

class SparseBoolMat(object):
    """
    Wraps a scipy COO sparse matrix so it can be used as a boolean matrix.
    """
    TRUE_SCALAR = sparse.coo_matrix(([True], ([0], [0])),
                                    shape=(1, 1), dtype='bool')
    FALSE_SCALAR = sparse.coo_matrix(([], ([], [])),
                                     shape=(1, 1), dtype='bool')
    def __init__(self, value):
        """
        value - the underlying COO sparse matrix.
        """
        self.value = value

    def reduce(self):
        """
        If the matrix in self is empty or full, then returns
        a SparseBoolMat containing a 1x1 matrix.
        """
        if self == True:
            return SparseBoolMat.TRUE_MAT
        elif self == False:
            return SparseBoolMat.FALSE_MAT
        else:
            return self

    def any(self):
        """
        Returns whether any of the entries are non-zero.
        """
        return self.value.nnz != 0

    def all(self):
        """
        Returns whether all the entries non-zero.
        """
        return self.value.nnz == len(self)

    @reduce_result
    def __getitem__(self, key):
        """
        Returns a SparseBoolMat containing an index/slice
        of the matrix in self.

        key - the index/slice into the matrix.
        """
        value = self.value.tolil()
        value = value[key]
        return SparseBoolMat(value.tocoo())

    def __len__(self):
        """
        Returns the number of entries in self's matrix.
        """
        return self.value.shape[0]*self.value.shape[1]

    @staticmethod
    def short_circuit(lh_mat, rh_mat, short_circuit_bool, short_circuit_mat):
        """
        Abstracts the logic of __or__ and __and__ when one argument is an empty
        or full matrix.
        Returns None if neither argument was empty or full.

        lh_mat, rh_mat - the SparseBoolMat arguments to __or__ or __and__.
        short_circuit_bool - the bool argument that would short circuit
                             the evaluation of the binary operator.
        short_circuit_mat - the matrix that the binary operator should
                            evaluate to if short circuited.
        """
        if lh_mat == short_circuit_bool or rh_mat == short_circuit_bool:
            return short_circuit_mat
        elif lh_mat == (not short_circuit_bool):
            return rh_mat
        elif rh_mat == (not short_circuit_bool):
            return lh_mat
        else:
            return None

    @reduce_result
    def __or__(self, other):
        """
        Takes the elementwise OR of the arguments.
        """
        result = SparseBoolMat.short_circuit(self, other,
                                             True, SparseBoolMat.TRUE_MAT)
        if result is not None:
            return result
        else:
            result = (self.value + other.value).astype('bool')
            return SparseBoolMat(result)

    def promote(self, size):
        """
        Promotes a SparseBoolMat containing a 1x1 matrix
        to a SparseBoolMat containing an empty or full
        matrix of the desired size.

        Has no effect on a SparseBoolMat containing a matrix
        that's not 1x1.

        size - the desired dimensions of the matrix.
        """
        if size != (1, 1) and len(self) == 1:
            # Promote true to a full matrix.
            if self == True:
                mat = np.empty(size)
                mat.fill(True)
                mat = sparse.coo_matrix(mat)
            # Promote False to an empty matrix.
            else:
                mat = sparse.coo_matrix(([], ([], [])),
                                        shape=size, dtype='bool')
            return SparseBoolMat(mat)
        else:
            return self

    @reduce_result
    def __mul__(self, other):
        """
        Multiplies the arguments.
        """
        result = self.value.dot(other.value).astype('bool')
        return SparseBoolMat(result)

    # For elementwise multiplication/bitwise and.
    @reduce_result
    def __and__(self, other):
        """
        Takes the elementwise AND of the arguments.
        """
        result = SparseBoolMat.short_circuit(self, other,
                                             False, SparseBoolMat.FALSE_MAT)
        if result is not None:
            return result
        else:
            result = self.value.multiply(other.value).astype('bool')
            return SparseBoolMat(result)

    def __eq__(self, other):
        """
        If other is a bool, returns whether the matrix in self
        can be reduced to that bool (i.e., is empty or full).

        Otherwise returns whether the matrices in self
        and other have the same entries.
        """
        if isinstance(other, bool):
            if other:
                return self.all()
            else:
                return not self.any()
        self = self.reduce()
        other = other.reduce()
        return self.value.shape == other.value.shape and \
               abs(self.value - other.value).nnz == 0

    def __str__(self):
        """
        Returns the to string value of the underlying matrix.
        """
        return str(self.value)

    def __repr__(self):
        """
        Returns the string representation of the underlying matrix.
        """
        return self.value.__repr__()

    @property
    def T(self):
        """
        Returns a SparseBoolMat containing the transpose
        of the matrix in self.
        """
        return SparseBoolMat(self.value.T)

# Scalar SparseBoolMat's for external use.
SparseBoolMat.TRUE_MAT = SparseBoolMat(SparseBoolMat.TRUE_SCALAR)
SparseBoolMat.FALSE_MAT = SparseBoolMat(SparseBoolMat.FALSE_SCALAR)
