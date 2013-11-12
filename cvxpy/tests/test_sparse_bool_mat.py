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

from cvxpy.utilities import SparseBoolMat
from nose.tools import *
import numpy as np
from scipy import sparse

class TestSparseBoolMat(object):
    """ Unit tests for the utilities/SparseBoolMat class. """
    @classmethod
    def setup_class(self):
        # Scalar matrix.
        self.scalar = SparseBoolMat(sparse.coo_matrix(([True],([0],[0])),shape=(1,1)))

        self.n = 4
        # Vectors.
        self.arr = np.array(self.n*[True])
        self.mixed_arr = np.array(self.n/2*[True, False])
        self.true_vec = SparseBoolMat(sparse.coo_matrix(self.arr))
        self.false_vec = SparseBoolMat(sparse.coo_matrix(~self.arr))

        # Dense matrices.
        self.mat = np.vstack(self.n*[self.arr])
        self.mixed = np.vstack(self.n*[self.mixed_arr])
        self.true_mat = SparseBoolMat(sparse.coo_matrix(self.mat))
        self.false_mat = SparseBoolMat(sparse.coo_matrix(~self.mat))
        self.mixed_mat = SparseBoolMat(sparse.coo_matrix(self.mixed))

        # Diagonal matrices.
        self.spdiag = sparse.eye(self.n, self.n).astype('bool')
        self.diag_spmat = SparseBoolMat(self.spdiag)

        # Reverse diag COO sparse matrix.
        vals = range(self.n)
        I = np.array(vals)
        vals.reverse()
        J = np.array(vals)
        V = self.arr
        self.coo = sparse.coo_matrix((V,(I,J)), shape=(self.n, self.n))
        self.coo_spmat = SparseBoolMat(self.coo)

        # X pattern sparse matrix.
        vals = range(self.n)
        I = np.array(vals + vals)
        vals.reverse()
        J = np.array(vals + range(self.n))
        V = np.array(2*self.n*[True])
        self.x = sparse.coo_matrix((V,(I,J)), shape=(self.n, self.n))
        self.x_spmat = SparseBoolMat(self.x)

        # Empty and full matrices.
        self.false_mat = SparseBoolMat.FALSE_MAT.promote((self.n, self.n))
        self.true_mat = SparseBoolMat.TRUE_MAT.promote((self.n, self.n))

    def test_or(self):
        """
        Test the | operator.
        """
        assert_equals(self.diag_spmat | self.diag_spmat, self.diag_spmat)
        assert_equals(self.diag_spmat | SparseBoolMat.FALSE_MAT, self.diag_spmat)
        assert_equals(self.diag_spmat | SparseBoolMat.TRUE_MAT, SparseBoolMat.TRUE_MAT)
        assert_equals(SparseBoolMat.FALSE_MAT | self.diag_spmat, self.diag_spmat)
        assert_equals(self.diag_spmat | self.coo_spmat, self.x_spmat)

    def test_and(self):
        """
        Test the & operator.
        """
        assert_equals(self.diag_spmat & self.diag_spmat, self.diag_spmat)
        assert_equals(self.diag_spmat & SparseBoolMat.FALSE_MAT, SparseBoolMat.FALSE_MAT)
        assert_equals(self.diag_spmat & SparseBoolMat.TRUE_MAT, self.diag_spmat)
        assert_equals(SparseBoolMat.FALSE_MAT & self.diag_spmat, SparseBoolMat.FALSE_MAT)
        assert_equals(self.x_spmat & self.coo_spmat, self.coo_spmat)
        assert_equals(self.diag_spmat & self.coo_spmat, SparseBoolMat.FALSE_MAT)

    def test_mul(self):
        """
        Test the * operator.
        """
        assert_equals(self.x_spmat * self.x_spmat, self.x_spmat)
        assert_equals(self.diag_spmat * self.coo_spmat, self.coo_spmat)
        assert_equals(self.diag_spmat * self.true_mat, SparseBoolMat.TRUE_MAT)
        assert_equals(self.false_mat * self.diag_spmat, SparseBoolMat.FALSE_MAT)
        assert_equals(self.x_spmat * self.coo_spmat, self.x_spmat)
        assert_equals(self.x_spmat * self.false_mat, SparseBoolMat.FALSE_MAT)

    def test_any(self):
        """
        Test the any operator.
        """
        assert self.diag_spmat.any()
        assert self.coo_spmat.any()

    def test_promote(self):
        """
        Test matrix promotion.
        """
        assert_equals(self.diag_spmat.promote((5,5)), self.diag_spmat)
        mat = SparseBoolMat.FALSE_MAT.promote((5,4))
        assert_equals(mat.value.shape, (5,4))
        mat = SparseBoolMat.TRUE_MAT.promote((1,1))
        assert_equals(mat, SparseBoolMat.TRUE_MAT)

    def test_transpose(self):
        """
        Test the transpose method.
        """
        assert_equals(self.diag_spmat.T, self.diag_spmat)
        assert_equals(SparseBoolMat.FALSE_MAT.T, SparseBoolMat.FALSE_MAT)
        size = self.mixed_mat.value.shape
        assert_equals(self.mixed_mat.T.value.shape, (size[1], size[0]))

    def test_vstack(self):
        """
        Test the vstack method.
        """
        mat = SparseBoolMat.vstack(self.diag_spmat)
        assert_equals(mat, self.diag_spmat)
        mat = SparseBoolMat.vstack(self.diag_spmat, self.x_spmat)
        assert_equals(mat[0:self.n, :], self.diag_spmat)
        assert_equals(mat[self.n:, :], self.x_spmat)
