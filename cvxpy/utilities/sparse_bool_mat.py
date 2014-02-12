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
        return self.value.nnz == self.size

    def __getitem__(self, key):
        """Indexes/slices into self's matrix.

        Args:
            key: The index/slice into self's matrix.

        Returns:
            A SparseBoolMat containing an index/slice of self's matrix.
        """
        value = self.value.tolil()
        value = value[key]
        if np.isscalar(value):
            return value
        else:
            return SparseBoolMat(value.tocoo())

    @property
    def shape(self):
        """The dimensions of the internal matrix.
        """
        return self.value.shape

    @property
    def size(self):
        """The number of entries in the internal matrix.
        """
        return self.value.shape[0]*self.value.shape[1]

    def __or__(self, other):
        """Elementwise OR between a SparseBoolMat and a bool scalar/matrix.

        Args:
            self: The left-hand SparseBoolMat.
            other: The right-hand bool scalar/matrix.

        Returns:
            The result of the elementwise OR.
        """
        # Short-circuit evaluation for bools.
        if isinstance(other, (np.bool_, bool)):
            if other:
                return np.bool_(True)
            else:
                return self
        # Convert to ndarray if other is ndarray.
        elif isinstance(other, np.ndarray):
            return self.todense() | other
        else:
            result = (self.value + other.value).astype('bool_')
            return SparseBoolMat(result)

    def __ror__(self, other):
        return self | other

    def __and__(self, other):
        """Elementwise AND between a SparseBoolMat and a bool scalar/matrix.

        Args:
            self: The left-hand SparseBoolMat.
            other: The right-hand bool scalar/matrix.

        Returns:
            The result of the elementwise AND.
        """
        # Short-circuit evaluation for bools.
        if isinstance(other, (np.bool_, bool)):
            if not other:
                return np.bool_(False)
            else:
                return self
        # Convert to ndarray if other is ndarray.
        elif isinstance(other, np.ndarray):
            return self.todense() | other
        else:
            result = self.value.multiply(other.value).astype('bool_')
            return SparseBoolMat(result)

    def __rand__(self, other):
        return self & other

    def __mul__(self, other):
        """Matrix multiplication of two SparseBoolMats.

        Args:
            self: The left-hand SparseBoolMat.
            other: The right-hand SparseBoolMat.

        Returns:
            The product of the SparseBoolMats.
        """
        result = self.value.dot(other.value).astype('bool_')
        return SparseBoolMat(result)

    def __eq__(self, other):
        """Evaluates matrix equality.

        Empty matrices and full matrices are equal regardless of size.

        Args:
            other: A SparseBoolMat

        Returns:
            Whether the matrices are equal.
        """
        # Short-circuit evaluation for bools.
        if isinstance(other, (np.bool_)):
            if other:
                return self.all()
            else:
                return not self.any()
        # Convert to ndarray if other is ndarray.
        elif isinstance(other, np.ndarray):
            return self.todense() == other
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

    def todense(self):
        """Converts the SparseBoolMat to a Numpy ndarray.
        """
        # Must be int64 for todense().
        dense = self.value.astype('int64').todense()
        # Convert back to bool.
        return dense.astype('bool_')
