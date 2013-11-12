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

class SparseBoolMat(object):
    """Wraps a SciPy COO sparse matrix so it can be used as a boolean matrix.

    Attributes:
        value: The underlying COO sparse matrix.
    """

    def __init__(self, value):
        self.value = value

    def any(self):
        """Returns whether any of the entries are True.
        """
        return self.value.nnz != 0

    def all(self):
        """Returns whether all the entries are False.
        """
        return self.value.nnz == len(self)

    def __getitem__(self, key):
        """Indexes/slices into self's matrix.

        Args:
            key: The index/slice into self's matrix.

        Returns:
            A SparseBoolMat containing an index/slice of self's matrix.
        """
        value = self.value.tolil()
        value = value[key]
        return SparseBoolMat(value.tocoo())

    def __len__(self):
        """The number of entries in self's matrix.
        """
        return self.value.shape[0]*self.value.shape[1]

    def promote(self, size):
        """Promotes a 1x1 matrix to the desired size.

        Has no effect on a SparseBoolMat containing a matrix
        that's not 1x1.

        Args:
            size: The desired dimensions of the matrix.

        Returns:
            A SparseBoolMat
        """
        if size != (1, 1) and len(self) == 1:
            # Promote True to a full matrix.
            if self.all():
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

    @staticmethod
    def _promote_args(lh_mat, rh_mat):
        """Promotes the arguments to a common size.

        Args:
            lh_mat: A SparseBoolMat
            rh_mat: A SparseBoolMat

        Returns:
            A tuple of SparseBoolMats with a common size.
        """
        size = max(lh_mat.value.shape, rh_mat.value.shape)
        lh_mat = lh_mat.promote(size)
        rh_mat = rh_mat.promote(size)
        return (lh_mat, rh_mat)

    def __or__(self, other):
        """Elementwise OR

        Args:
            other: A SparseBoolMat

        Returns:
            A SparseBoolMat
        """
        self, other = SparseBoolMat._promote_args(self, other)
        result = (self.value + other.value).astype('bool')
        return SparseBoolMat(result)

    def __and__(self, other):
        """Elementwise AND

        Args:
            other: A SparseBoolMat

        Returns:
            A SparseBoolMat
        """
        self, other = SparseBoolMat._promote_args(self, other)
        result = self.value.multiply(other.value).astype('bool')
        return SparseBoolMat(result)


    def __mul__(self, other):
        """Matrix multiplication.

        Args:
            other: A SparseBoolMat

        Returns:
            A SparseBoolMat
        """
        result = self.value.dot(other.value).astype('bool')
        return SparseBoolMat(result)

    def __eq__(self, other):
        """Evaluates matrix equality.

        Empty matrices and full matrices are equal regardless of size.

        Args:
            other: A SparseBoolMat

        Returns:
            Whether the matrices are equal.
        """
        return (self.value.shape == other.value.shape and \
                abs(self.value - other.value).nnz == 0) or \
                self.all() and other.all() or \
                (not self.any() and not other.any())

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return "SparseBoolMat(%s)" % self.value.__repr__()

    @property
    def T(self):
        """Returns a SparseBoolMat containing the transpose of self's matrix.
        """
        return SparseBoolMat(self.value.T)

    @staticmethod
    def vstack(*args):
        """Vertically stacks the arguments into a single matrix.

        Args:
            *args: SparseBoolMats

        Returns:
            A SparseBoolMat
        """
        cols = args[0].value.shape[1]
        rows = sum([spmat.value.shape[0] for spmat in args])
        stacked = sparse.coo_matrix(([], ([], [])),
                                    shape=(rows, cols), dtype='bool')
        stacked = stacked.tolil()
        offset = 0
        for spmat in args:
            height = spmat.value.shape[1]
            stacked[offset:offset + height, :] = spmat.value
            offset += height
        return SparseBoolMat(stacked.tocoo())

# Scalar SparseBoolMat's for external use.
TRUE_SCALAR = sparse.coo_matrix(([True], ([0], [0])),
                                shape=(1, 1), dtype='bool')
FALSE_SCALAR = sparse.coo_matrix(([], ([], [])),
                                 shape=(1, 1), dtype='bool')
SparseBoolMat.TRUE_MAT = SparseBoolMat(TRUE_SCALAR)
SparseBoolMat.FALSE_MAT = SparseBoolMat(FALSE_SCALAR)
