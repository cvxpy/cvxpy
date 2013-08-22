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

from cvxpy.utilities import BoolMat
from cvxpy.utilities import SparseBoolMat
from nose.tools import *
import numpy as np
from scipy import sparse

class TestSparseBoolMat(object):
    """ Unit tests for the utilities/SparseBoolMat class. """
    @classmethod
    def setup_class(self):
        n = 4
        # Vectors.
        self.arr = np.array(n*[True])
        self.mixed_arr = np.array(n/2 * [True, False])
        self.true_vec = BoolMat(self.arr)
        self.false_vec = BoolMat(~self.arr)

        # Dense matrices.
        self.mat = np.vstack(n*[self.arr])
        self.mixed = np.vstack(n*[self.mixed_arr])
        self.true_mat = BoolMat(self.mat)
        self.false_mat = BoolMat(~self.mat)
        self.mixed_mat = BoolMat(self.mixed)

        # Diagonal matrices.
        self.diag = np.eye(n).astype('bool')
        self.diag_mat = BoolMat(self.diag)
        self.spdiag = sparse.eye(n).astype('bool')
        self.diag_spmat = SparseBoolMat(self.spdiag)

        # Reverse diag COO sparse matrix.
        vals = range(n)
        I = np.array(vals)
        vals.reverse()
        J = np.array(vals)
        V = self.arr
        self.coo = sparse.coo_matrix((V,(I,J)),shape=(n,n))
        self.coo_spmat = SparseBoolMat(self.coo)

        # X pattern sparse matrix.
        vals = range(n)
        I = np.array(vals + vals)
        vals.reverse()
        J = np.array(vals + range(n))
        V = np.array(2*n*[True])
        self.x = sparse.coo_matrix((V,(I,J)),shape=(n,n))
        self.x_spmat = SparseBoolMat(self.x)

        # Empty sparse matrix.
        self.empty = sparse.coo_matrix(([],([],[])),shape=(n,n))
        self.empty_spmat = SparseBoolMat(self.empty)

    # Test the | operator.
    def test_or(self):
        assert_equals(self.diag_spmat | self.diag_spmat, self.diag_spmat)
        assert_equals(self.diag_spmat | self.false_mat, self.diag_mat)
        assert_equals(self.diag_spmat | True, True)
        assert_equals(False | self.diag_spmat, self.diag_spmat)
        assert_equals(self.diag_spmat | self.coo_spmat, self.x_spmat)

    # Test the & operator.
    def test_and(self):
        assert_equals(self.diag_spmat & self.diag_spmat, self.diag_spmat)
        assert_equals(self.diag_spmat & self.false_mat, self.false_mat)
        assert_equals(self.diag_spmat & True, self.diag_spmat)
        assert_equals(False & self.diag_spmat, False)
        assert_equals(self.x_spmat & self.coo_spmat, self.coo_spmat)
        assert_equals(self.diag_spmat & self.coo_spmat, self.empty_spmat)

    # Test the * operator.
    def test_mul(self):
        assert_equals(self.x_spmat * self.x_spmat, self.x_spmat)
        assert_equals(self.diag_spmat * self.coo_spmat, self.coo_spmat)
        assert_equals(self.diag_spmat * True, self.diag_spmat)
        assert_equals(False * self.diag_spmat, False)
        assert_equals(self.x_spmat * self.coo_spmat, self.x_spmat)
        assert_equals(self.x_spmat * self.empty_spmat, self.empty_spmat)

    # Test the any operator.
    def test_any(self):
        assert self.diag_spmat.any()
        assert self.coo_spmat.any()
        assert not self.empty_spmat.any()