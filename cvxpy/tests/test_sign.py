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

from cvxpy.utilities import Sign
from cvxpy.utilities import SparseBoolMat
from nose.tools import *
import numpy as np
from scipy import sparse

class TestSign(object):
    """ Unit tests for the utilities/sign class. """
    @classmethod
    def setup_class(self):
        self.n = 5
        self.arr = np.array(self.n*[[True]])
        true_vec = sparse.coo_matrix(self.arr)
        false_vec = sparse.coo_matrix(~self.arr)
        # Vectors
        self.neg_vec = Sign(SparseBoolMat(true_vec), SparseBoolMat(false_vec))
        self.pos_vec = Sign(SparseBoolMat(false_vec), SparseBoolMat(true_vec))
        self.unknown_vec = Sign(SparseBoolMat(true_vec), SparseBoolMat(true_vec))
        self.zero_vec = Sign(SparseBoolMat(false_vec), SparseBoolMat(false_vec))

        # Dense Matrices
        self.mat = sparse.coo_matrix(np.vstack(self.n*[self.arr]))
        self.neg_mat = sparse.coo_matrix(~np.vstack(self.n*[self.arr]))
        self.neg_mat = Sign(SparseBoolMat(self.mat), SparseBoolMat(self.neg_mat))
        self.pos_mat = Sign(SparseBoolMat(self.neg_mat), SparseBoolMat(self.mat))
        self.unknown_mat = Sign(SparseBoolMat(self.mat), SparseBoolMat(self.mat))
        self.zero_mat = Sign(SparseBoolMat(self.neg_mat), SparseBoolMat(self.neg_mat))

        # Sparse Matrices
        self.spmat = sparse.eye(self.n, self.n).astype('bool')
        self.neg_spmat = Sign(SparseBoolMat(self.spmat), False)
        self.pos_spmat = Sign(False, SparseBoolMat(self.spmat))
        self.unknown_spmat = Sign(SparseBoolMat(self.spmat), SparseBoolMat(self.spmat))

    # Scalar sign tests.
    def test_add(self):
        assert_equals(Sign.POSITIVE + Sign.NEGATIVE, Sign.UNKNOWN)
        assert_equals(Sign.NEGATIVE + Sign.ZERO, Sign.NEGATIVE)
        assert_equals(Sign.POSITIVE + Sign.POSITIVE, Sign.POSITIVE)
        assert_equals(Sign.UNKNOWN + Sign.ZERO, Sign.UNKNOWN)

    def test_sub(self):
        assert_equals(Sign.POSITIVE - Sign.NEGATIVE, Sign.POSITIVE)
        assert_equals(Sign.NEGATIVE - Sign.ZERO, Sign.NEGATIVE)
        assert_equals(Sign.POSITIVE - Sign.POSITIVE, Sign.UNKNOWN)

    def test_mul(self):
        assert_equals(Sign.ZERO * Sign.POSITIVE, Sign.ZERO)
        assert_equals(Sign.UNKNOWN * Sign.POSITIVE, Sign.UNKNOWN)
        assert_equals(Sign.POSITIVE * Sign.NEGATIVE, Sign.NEGATIVE)
        assert_equals(Sign.NEGATIVE * Sign.NEGATIVE, Sign.POSITIVE)
        assert_equals(Sign.ZERO * Sign.UNKNOWN, Sign.ZERO)

    def test_neg(self):
        assert_equals(-Sign.ZERO, Sign.ZERO)
        assert_equals(-Sign.POSITIVE, Sign.NEGATIVE)

    # Dense sign matrix tests.
    def test_dmat_add(self):
        assert_equals(self.pos_vec + self.neg_vec, self.unknown_vec)
        result = Sign(True, true_vec) # Reduced to scalar
        assert_equals(self.pos_vec + Sign.NEGATIVE, result)
        assert_equals(self.neg_vec + self.zero_vec, self.neg_vec)
        result = Sign(True, BoolMat(~self.mat)) # Reduced to scalar
        assert_equals(self.neg_mat + Sign.NEGATIVE, result)
        assert_equals(self.pos_vec + self.pos_vec, self.pos_vec)
        assert_equals(self.unknown_mat + self.zero_mat, self.unknown_mat)
        assert_equals(Sign.UNKNOWN + self.pos_mat, Sign.UNKNOWN)

    def test_dmat_sub(self):
        assert_equals(self.pos_vec - self.neg_vec, self.pos_vec)
        result = Sign(BoolMat(false_mat), True) # Reduced to scalar
        assert_equals(Sign.POSITIVE - self.neg_vec, result)
        assert_equals(self.neg_vec - self.zero_vec, self.neg_vec)
        assert_equals(self.pos_mat - self.pos_mat, self.unknown_mat)
        assert_equals(self.zero_vec - Sign.UNKNOWN, Sign.UNKNOWN)

    def test_dmat_mult(self):
        assert_equals(Sign.mul(Sign.ZERO, (1,1), self.pos_vec, (self.n,1)), Sign.ZERO)
        assert_equals(Sign.mul(self.unknown_vec, (self.n,1), Sign.POSITIVE, (1,1)), self.unknown_vec)
        assert_equals(Sign.mul(self.pos_vec, (self.n,1), Sign.NEGATIVE, (1,1)), self.neg_vec)
        assert_equals(Sign.mul(self.neg_mat, (self.n,self.n),
                               self.neg_vec, (self.n,1)), self.pos_vec)
        assert_equals(Sign.mul(self.zero_mat, (self.n,self.n),
                               self.unknown_vec, (self.n,1)), self.zero_vec)
        assert_equals(Sign.mul(self.neg_mat, (self.n,self.n),
                               self.pos_mat, (self.n,self.n)), self.neg_mat)
        assert_equals(Sign.mul(self.unknown_mat, (self.n,self.n),
                               self.pos_mat, (self.n,self.n)), self.unknown_mat)

    def test_dmat_neg(self):
        assert_equals(-self.zero_vec, self.zero_vec)
        assert_equals(-self.pos_vec, self.neg_vec)
        assert_equals(-self.neg_mat, self.pos_mat)

    # Sparse sign matrix tests.
    def test_sparse_add(self):
        assert_equals(self.pos_spmat + self.neg_spmat, self.unknown_spmat)
        assert_equals(self.neg_spmat + Sign.ZERO, self.neg_spmat)
        assert_equals(self.pos_spmat + self.pos_spmat, self.pos_spmat)
        assert_equals(self.unknown_mat + self.neg_mat, self.unknown_mat)
        result = Sign(self.true_mat, SparseBoolMat(self.spmat))
        assert_equals(self.pos_spmat + Sign.NEGATIVE, result)
        assert_equals(self.neg_spmat + self.unknown_mat, self.unknown_mat)
        assert_equals(self.pos_spmat + self.pos_mat, self.pos_mat)

    def test_sparse_sub(self):
        assert_equals(self.pos_spmat - self.neg_spmat, self.pos_spmat)
        assert_equals(Sign.POSITIVE - self.neg_spmat, Sign.POSITIVE)
        assert_equals(self.neg_spmat - self.unknown_mat, self.unknown_mat)
        assert_equals(self.neg_spmat - self.neg_mat,
                      self.unknown_spmat + self.pos_mat)
        assert_equals(self.neg_spmat - self.pos_spmat, self.neg_spmat)

    def test_sparse_mult(self):
        assert_equals(Sign.ZERO * self.pos_spmat, Sign.ZERO)
        assert_equals(self.unknown_spmat * Sign.POSITIVE, self.unknown_spmat)
        assert_equals(self.pos_spmat * Sign.NEGATIVE, self.neg_spmat)
        assert_equals(self.neg_mat * self.neg_spmat, self.pos_mat)
        assert_equals(self.zero_mat * self.unknown_spmat, self.zero_mat)
        assert_equals(self.neg_spmat * self.pos_spmat, self.neg_spmat)
        assert_equals(self.unknown_spmat * self.pos_spmat, self.unknown_spmat)

        # Asymmetric multiplication.
        m = 2
        fat = np.vstack(m*[self.arr])
        fat_pos = Sign(False, BoolMat(fat))
        assert_equals(Sign.mul(fat_pos, (1,1), self.pos_spmat, (1,1)), fat_pos)

    def test_sparse_neg(self):
        assert_equals(-self.unknown_spmat, self.unknown_spmat)
        assert_equals(-self.pos_spmat, self.neg_spmat)
        assert_equals(-self.neg_spmat, self.pos_spmat)

    # Tests the promote method.
    def test_promote(self):
        sign = Sign.POSITIVE.promote((3,4))
        assert_equals(sign.neg_mat.value.shape, (3,4))
        assert_equals(sign.pos_mat.value.shape, (3,4))

    # Tests the is_positive and is_negative methods.
    def test_is_sign(self):
        assert Sign.POSITIVE.is_positive()
        assert not Sign.NEGATIVE.is_positive()
        assert not Sign.UNKNOWN.is_positive()
        assert Sign.ZERO.is_positive()

        assert not Sign.POSITIVE.is_negative()
        assert Sign.NEGATIVE.is_negative()
        assert not Sign.UNKNOWN.is_negative()
        assert Sign.ZERO.is_negative()