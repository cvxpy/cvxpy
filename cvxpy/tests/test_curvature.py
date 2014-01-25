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
        self.arr = np.atleast_2d(self.n*[[np.bool_(True)]])
        # Vectors
        self.cvx_vec = Curvature(self.arr, ~self.arr, np.bool_(True))
        self.conc_vec = Curvature(~self.arr, self.arr, np.bool_(True))
        self.noncvx_vec = Curvature(self.arr, self.arr, np.bool_(True))
        self.aff_vec = Curvature(~self.arr, ~self.arr, np.bool_(True))
        self.const_vec = Curvature(~self.arr, ~self.arr, np.bool_(False))

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
                      Curvature.UNKNOWN), Curvature.CONSTANT)

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
        curv = Curvature.CONSTANT.promote(3, 4)
        assert_equals(curv.cvx_mat.shape, (3, 4))
        assert_equals(curv.conc_mat.shape, (3, 4))

    def test_get_readable_repr(self):
        """Tests the get_readable_repr method.
        """
        assert_equals(Curvature.CONSTANT.get_readable_repr(1,1), Curvature.CONSTANT_KEY)
        assert_equals(Curvature.CONSTANT.get_readable_repr(5,4), Curvature.CONSTANT_KEY)

        assert_equals(Curvature.AFFINE.get_readable_repr(1,1), Curvature.AFFINE_KEY)
        assert_equals(Curvature.AFFINE.get_readable_repr(5,4), Curvature.AFFINE_KEY)

        assert_equals(Curvature.CONVEX.get_readable_repr(1,1), Curvature.CONVEX_KEY)
        assert_equals(Curvature.CONVEX.get_readable_repr(5,4), Curvature.CONVEX_KEY)

        assert_equals(Curvature.CONCAVE.get_readable_repr(1,1), Curvature.CONCAVE_KEY)
        assert_equals(Curvature.CONCAVE.get_readable_repr(5,4), Curvature.CONCAVE_KEY)

        assert_equals(Curvature.UNKNOWN.get_readable_repr(1,1), Curvature.UNKNOWN_KEY)
        assert_equals(Curvature.UNKNOWN.get_readable_repr(5,4), Curvature.UNKNOWN_KEY)

        # Mixed curvatures.
        mix_vec = np.vstack([self.arr, ~self.arr])
        cv = Curvature(mix_vec, mix_vec, np.bool_(True))
        unknown_str_arr = np.atleast_2d(self.n*[[Curvature.UNKNOWN_KEY]])
        affine_str_arr = np.atleast_2d(self.n*[[Curvature.AFFINE_KEY]])
        str_arr = np.vstack([unknown_str_arr, affine_str_arr])
        assert (cv.get_readable_repr(2*self.n, 1) == str_arr).all()

        cv = Curvature(mix_vec, ~mix_vec, np.bool_(True))
        conc_str_arr = np.atleast_2d(self.n*[[Curvature.CONCAVE_KEY]])
        cvx_str_arr = np.atleast_2d(self.n*[[Curvature.CONVEX_KEY]])
        str_arr = np.vstack([cvx_str_arr, conc_str_arr])
        assert (cv.get_readable_repr(2*self.n, 1) == str_arr).all()

    # # Test the vstack method.
    # def test_vstack(self):
    #     curvs = self.n*[(Curvature.CONSTANT, (1,1))]
    #     vs = Curvature.vstack(*curvs)
    #     print vs
    #     assert_equals(vs, self.const_vec)

    #     curvs = self.n*[(Curvature.CONVEX, (1,1))]
    #     assert_equals(Curvature.vstack(*curvs), self.cvx_vec)