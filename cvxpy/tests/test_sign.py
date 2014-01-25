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
        self.arr = np.atleast_2d(self.n*[np.bool_(True)]).T
        # Vectors
        self.neg_vec = Sign(self.arr, ~self.arr)
        self.pos_vec = Sign(~self.arr, self.arr)
        self.unknown_vec = Sign(self.arr, self.arr)
        self.zero_vec = Sign(~self.arr, ~self.arr)

        # Dense Matrices
        self.true_mat = np.hstack(self.n*[self.arr])
        self.neg_mat = Sign(self.true_mat, ~self.true_mat)
        self.pos_mat = Sign(~self.true_mat, self.true_mat)
        self.unknown_mat = Sign(self.true_mat, self.true_mat)
        self.zero_mat = Sign(~self.true_mat, ~self.true_mat)

        # Sparse Matrices
        self.spmat = sparse.eye(self.n, self.n).astype('bool_')
        self.neg_spmat = Sign(SparseBoolMat(self.spmat), np.bool_(False))
        self.pos_spmat = Sign(np.bool_(False), SparseBoolMat(self.spmat))
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
        result = Sign(np.bool_(True), self.arr)
        assert_equals(self.pos_vec + Sign.NEGATIVE, result)
        assert_equals(self.neg_vec + self.zero_vec, self.neg_vec)
        result = Sign(np.bool_(True), ~self.true_mat)
        assert_equals(self.neg_mat + Sign.NEGATIVE, result)
        assert_equals(self.pos_vec + self.pos_vec, self.pos_vec)
        assert_equals(self.unknown_mat + self.zero_mat, self.unknown_mat)
        assert_equals(Sign.UNKNOWN + self.pos_mat, Sign.UNKNOWN)

    def test_dmat_sub(self):
        assert_equals(self.pos_vec - self.neg_vec, self.pos_vec)
        result = Sign(~self.true_mat, np.bool_(True))
        assert_equals(Sign.POSITIVE - self.neg_vec, result)
        assert_equals(self.neg_vec - self.zero_vec, self.neg_vec)
        assert_equals(self.pos_mat - self.pos_mat, self.unknown_mat)
        assert_equals(self.zero_vec - Sign.UNKNOWN, Sign.UNKNOWN)

    def test_dmat_mult(self):
        assert_equals(Sign.ZERO.promote(1, self.n) * self.pos_vec, Sign.ZERO)
        assert_equals(self.unknown_vec * Sign.POSITIVE, self.unknown_vec)
        assert_equals(self.pos_vec * Sign.NEGATIVE, self.neg_vec)
        assert_equals(self.neg_mat * self.neg_vec, self.pos_vec)
        assert_equals(self.zero_mat * self.unknown_vec, self.zero_vec)
        assert_equals(self.neg_mat * self.pos_mat, self.neg_mat)
        assert_equals(self.unknown_mat * self.pos_mat, self.unknown_mat)

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
        size = (self.n, self.n)
        assert_equals(Sign.ZERO.promote(*size) * self.pos_spmat.promote(*size),
                      Sign.ZERO.promote(*size))
        assert_equals(self.unknown_spmat * Sign.POSITIVE, self.unknown_spmat)
        assert_equals(self.pos_spmat.promote(*size) * Sign.NEGATIVE, self.neg_spmat)
        assert_equals(self.neg_mat * self.neg_spmat.promote(*size), self.pos_mat)
        assert_equals(self.zero_mat.promote(*size) * self.unknown_spmat, self.zero_mat)
        assert_equals(self.neg_spmat.promote(*size) * self.pos_spmat.promote(*size), self.neg_spmat)
        assert_equals(self.unknown_spmat * self.pos_spmat.promote(*size), self.unknown_spmat)

        # Asymmetric multiplication.
        m = 2
        fat = np.hstack(m*[self.arr]).T
        fat_pos = Sign(np.bool_(False), fat)
        fat_size = (m, self.n)
        assert_equals(fat_pos.promote(*fat_size) * self.pos_spmat.promote(*size),
                      fat_pos.promote(*fat_size))

    def test_sparse_neg(self):
        assert_equals(-self.unknown_spmat, self.unknown_spmat)
        assert_equals(-self.pos_spmat, self.neg_spmat)
        assert_equals(-self.neg_spmat, self.pos_spmat)

    # Tests the promote method.
    def test_promote(self):
        sign = Sign.POSITIVE.promote(3, 4)
        assert_equals(sign.neg_mat.shape, (3, 4))
        assert_equals(sign.pos_mat.shape, (3, 4))

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

        assert Sign.ZERO.is_zero()
        assert not Sign.NEGATIVE.is_zero()

    def test_get_readable_repr(self):
        """Tests the get_readable_repr method.
        """
        assert_equals(Sign.POSITIVE.get_readable_repr(1,1), Sign.POSITIVE_KEY)
        assert_equals(Sign.POSITIVE.get_readable_repr(5,4), Sign.POSITIVE_KEY)

        assert_equals(Sign.NEGATIVE.get_readable_repr(1,1), Sign.NEGATIVE_KEY)
        assert_equals(Sign.NEGATIVE.get_readable_repr(5,4), Sign.NEGATIVE_KEY)

        assert_equals(Sign.ZERO.get_readable_repr(1,1), Sign.ZERO_KEY)
        assert_equals(Sign.ZERO.get_readable_repr(5,4), Sign.ZERO_KEY)

        assert_equals(Sign.UNKNOWN.get_readable_repr(1,1), Sign.UNKNOWN_KEY)
        assert_equals(Sign.UNKNOWN.get_readable_repr(5,4), Sign.UNKNOWN_KEY)

        # Mixed signs.
        mix_vec = np.vstack([self.arr, ~self.arr])
        cv = Sign(mix_vec, mix_vec)
        unknown_str_arr = np.atleast_2d(self.n*[[Sign.UNKNOWN_KEY]])
        zero_str_arr = np.atleast_2d(self.n*[[Sign.ZERO_KEY]])
        str_arr = np.vstack([unknown_str_arr, zero_str_arr])
        assert (cv.get_readable_repr(2*self.n, 1) == str_arr).all()

        cv = Sign(mix_vec, ~mix_vec)
        neg_str_arr = np.atleast_2d(self.n*[[Sign.NEGATIVE_KEY]])
        pos_str_arr = np.atleast_2d(self.n*[[Sign.POSITIVE_KEY]])
        str_arr = np.vstack([neg_str_arr, pos_str_arr])
        assert (cv.get_readable_repr(2*self.n, 1) == str_arr).all()
