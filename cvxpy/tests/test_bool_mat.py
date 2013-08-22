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
from nose.tools import *
import numpy as np

class TestBoolMat(object):
    """ Unit tests for the utilities/BoolMat class. """
    @classmethod
    def setup_class(self):
        n = 4
        # Vectors.
        self.arr = np.array(n*[True])
        self.mixed_arr = np.array(n/2 * [True, False])
        self.true_vec = BoolMat(self.arr)
        self.false_vec = BoolMat(~self.arr)

        # Matrices.
        self.mat = np.vstack(n*[self.arr])
        self.mixed = np.vstack(n*[self.mixed_arr])
        self.true_mat = BoolMat(self.mat)
        self.false_mat = BoolMat(~self.mat)
        self.mixed_mat = BoolMat(self.mixed)

    # Test the | operator.
    def test_or(self):
        assert_equals(self.false_mat | self.false_mat, self.false_mat)
        assert_equals(self.true_mat | self.false_mat, self.true_mat)
        assert_equals(self.true_mat | True, True)
        assert_equals(False | self.false_mat, self.false_mat)
        assert_equals(self.mixed_mat | self.true_mat, self.true_mat)
        assert_equals(self.false_mat | self.mixed_mat, self.mixed_mat)

    # Test the & operator.
    def test_and(self):
        assert_equals(self.true_mat & self.true_mat, self.true_mat)
        assert_equals(self.true_mat & self.false_mat, self.false_mat)
        assert_equals(self.true_mat & True, self.true_mat)
        assert_equals(False & self.false_mat, False)
        assert_equals(self.mixed_mat & self.true_mat, self.mixed_mat)
        assert_equals(self.false_mat & self.mixed_mat, self.false_mat)
        assert_equals(BoolMat(~self.mixed) & self.mixed_mat, self.false_mat)

    # Test the * operator.
    def test_mul(self):
        assert_equals(self.true_mat * self.true_mat, self.true_mat)
        assert_equals(self.true_mat * self.false_vec, self.false_vec)
        assert_equals(self.false_mat * True, self.false_mat)
        assert_equals(False * self.true_mat, False)
        assert_equals(self.mixed_mat * self.true_vec, self.true_vec)
        assert_equals(self.mixed_mat * self.mixed_mat, self.mixed_mat)

    # Test the any operator.
    def test_any(self):
        assert self.true_mat.any()
        assert not self.false_mat.any()
        assert self.mixed_mat.any()