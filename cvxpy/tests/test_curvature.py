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
from cvxpy.utilities import Curvature
from cvxpy.utilities import Sign
from nose.tools import assert_equals
import numpy as np
from scipy import sparse

class TestCurvature(object):
    """ Unit tests for the utilities/curvature class. """
    @classmethod
    def setup_class(self):
        self.n = 5
        self.arr = np.array(self.n*[[True]])
        true_mat = sparse.coo_matrix(self.arr)
        false_mat = sparse.coo_matrix(~self.arr)
        # Vectors
        self.cvx_vec = Curvature(true_mat, false_mat, False)
        self.conc_vec = Curvature(false_mat, true_mat, False)
        self.noncvx_vec = Curvature(true_mat, true_mat, False)
        self.aff_vec = Curvature(false_mat, false_mat, False)
        self.const_vec = Curvature(false_mat, false_mat, True)

    # TODO tests with matrices.
    def test_add(self):
        assert_equals(Curvature.CONSTANT + Curvature.CONVEX, Curvature.CONVEX)
        assert_equals(Curvature.UNKNOWN + Curvature.CONCAVE, Curvature.UNKNOWN)
        assert_equals(Curvature.CONVEX + Curvature.CONCAVE, Curvature.UNKNOWN)
        assert_equals(Curvature.CONVEX + Curvature.CONVEX, Curvature.CONVEX)
        assert_equals(Curvature.AFFINE + Curvature.CONCAVE, Curvature.CONCAVE)

    def test_sub(self):
        assert_equals(Curvature.CONSTANT - Curvature.CONVEX, Curvature.CONCAVE)
        assert_equals(Curvature.UNKNOWN - Curvature.CONCAVE, Curvature.UNKNOWN)
        assert_equals(Curvature.CONVEX - Curvature.CONCAVE, Curvature.CONVEX)
        assert_equals(Curvature.CONVEX - Curvature.CONVEX, Curvature.UNKNOWN)
        assert_equals(Curvature.AFFINE - Curvature.CONCAVE, Curvature.CONVEX)

    def test_sign_mult(self):
        assert_equals(Curvature.sign_mul(Sign.POSITIVE, 
                      Curvature.CONVEX), Curvature.CONVEX)
        assert_equals(Curvature.sign_mul(Sign.UNKNOWN, 
                      Curvature.CONSTANT), Curvature.CONSTANT)
        assert_equals(Curvature.sign_mul(Sign.NEGATIVE, 
                      Curvature.CONCAVE), Curvature.CONVEX)
        assert_equals(Curvature.sign_mul(Sign.ZERO, 
                      Curvature.UNKNOWN), Curvature.AFFINE)

    def test_neg(self):
        assert_equals(-Curvature.CONVEX, Curvature.CONCAVE)
        assert_equals(-Curvature.AFFINE, Curvature.AFFINE)

    # Tests the is_affine, is_convex, is_concave, and is_dcp methods
    def test_is_curvature(self):
        assert Curvature.CONSTANT.is_affine()
        assert Curvature.AFFINE.is_affine()
        assert not Curvature.CONVEX.is_affine()
        assert not Curvature.CONCAVE.is_affine()
        assert not Curvature.UNKNOWN.is_affine()

        assert Curvature.CONSTANT.is_convex()
        assert Curvature.AFFINE.is_convex()
        assert Curvature.CONVEX.is_convex()
        assert not Curvature.CONCAVE.is_convex()
        assert not Curvature.UNKNOWN.is_convex()

        assert Curvature.CONSTANT.is_concave()
        assert Curvature.AFFINE.is_concave()
        assert not Curvature.CONVEX.is_concave()
        assert Curvature.CONCAVE.is_concave()
        assert not Curvature.UNKNOWN.is_concave()

        assert Curvature.CONSTANT.is_dcp()
        assert Curvature.AFFINE.is_dcp()
        assert Curvature.CONVEX.is_dcp()
        assert Curvature.CONCAVE.is_dcp()
        assert not Curvature.UNKNOWN.is_dcp()

    # Tests the promote method.
    def test_promote(self):
        curv = Curvature.CONSTANT.promote((3,4))
        assert_equals(curv.cvx_mat.value.shape, (3,4))
        assert_equals(curv.conc_mat.value.shape, (3,4))

    # # Test the vstack method.
    # def test_vstack(self):
    #     curvs = self.n*[(Curvature.CONSTANT, (1,1))]
    #     vs = Curvature.vstack(*curvs)
    #     print vs
    #     assert_equals(vs, self.const_vec)

    #     curvs = self.n*[(Curvature.CONVEX, (1,1))]
    #     assert_equals(Curvature.vstack(*curvs), self.cvx_vec)