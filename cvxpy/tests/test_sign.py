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
from cvxpy.utilities import SignCurvMat
from nose.tools import *
import numpy as np

class TestSign(object):
    """ Unit tests for the expression/sign class. """
    @classmethod
    def setup_class(self):
        n = 5
        arr = np.array(n*[True])
        # Vectors
        self.neg_vec = Sign(SignCurvMat(arr), SignCurvMat(~arr))
        self.pos_vec = Sign(SignCurvMat(~arr), SignCurvMat(arr))
        self.unknown_vec = Sign(SignCurvMat(arr), SignCurvMat(arr))
        self.zero_vec = Sign(SignCurvMat(~arr), SignCurvMat(~arr))

        # Matrices
        mat = np.vstack(n*[arr])
        self.neg_mat = Sign(SignCurvMat(mat), SignCurvMat(~mat))
        self.pos_mat = Sign(SignCurvMat(~mat), SignCurvMat(mat))
        self.unknown_mat = Sign(SignCurvMat(mat), SignCurvMat(mat))
        self.zero_mat = Sign(SignCurvMat(~mat), SignCurvMat(~arr))
    
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

    def test_mult(self):
        assert_equals(Sign.ZERO * Sign.POSITIVE, Sign.ZERO)
        assert_equals(Sign.UNKNOWN * Sign.POSITIVE, Sign.UNKNOWN)
        assert_equals(Sign.POSITIVE * Sign.NEGATIVE, Sign.NEGATIVE)
        assert_equals(Sign.NEGATIVE * Sign.NEGATIVE, Sign.POSITIVE)
        assert_equals(Sign.ZERO * Sign.UNKNOWN, Sign.ZERO)

    def test_neg(self):
        assert_equals(-Sign.ZERO, Sign.ZERO)
        assert_equals(-Sign.POSITIVE, Sign.NEGATIVE)

    # Dense matrix sign tests.
    def test_dmat_add(self):
        assert_equals(self.pos_vec + self.neg_vec, self.unknown_vec)
        assert_equals(self.pos_vec + Sign.NEGATIVE, self.unknown_vec)
        assert_equals(self.neg_vec + self.zero_vec, self.neg_vec)
        assert_equals(self.neg_mat + Sign.NEGATIVE, self.neg_mat)
        assert_equals(self.pos_vec + self.pos_vec, self.pos_vec)
        assert_equals(self.unknown_mat + self.zero_mat, self.unknown_mat)

    def test_dmat_sub(self):
        assert_equals(self.pos_vec - self.neg_vec, self.pos_vec)
        assert_equals(Sign.POSITIVE - self.neg_vec, self.pos_vec)
        assert_equals(self.neg_vec - self.zero_vec, self.neg_vec)
        assert_equals(self.pos_mat - self.pos_mat, self.unknown_mat)
        assert_equals(self.zero_vec - Sign.UNKNOWN, self.unknown_vec)

    def test_dmat_mult(self):
        assert_equals(Sign.ZERO * self.pos_vec, self.zero_vec)
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