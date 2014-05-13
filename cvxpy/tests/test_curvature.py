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

from cvxpy.utilities import Curvature
from cvxpy.utilities import Sign
from nose.tools import assert_equals

class TestCurvature(object):
    """ Unit tests for the expression/curvature class. """
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
        assert_equals(Curvature.sign_mul(Sign.ZERO, Curvature.CONVEX), Curvature.CONSTANT)
        assert_equals(Curvature.sign_mul(Sign.NEGATIVE, Curvature.CONVEX), Curvature.CONCAVE)
        assert_equals(Curvature.sign_mul(Sign.NEGATIVE, Curvature.CONCAVE), Curvature.CONVEX)
        assert_equals(Curvature.sign_mul(Sign.NEGATIVE, Curvature.UNKNOWN), Curvature.UNKNOWN)
        assert_equals(Curvature.sign_mul(Sign.POSITIVE, Curvature.AFFINE), Curvature.AFFINE)
        assert_equals(Curvature.sign_mul(Sign.POSITIVE, Curvature.CONCAVE), Curvature.CONCAVE)
        assert_equals(Curvature.sign_mul(Sign.UNKNOWN, Curvature.CONSTANT), Curvature.CONSTANT)
        assert_equals(Curvature.sign_mul(Sign.UNKNOWN, Curvature.CONCAVE), Curvature.UNKNOWN)

    def test_neg(self):
        assert_equals(-Curvature.CONVEX, Curvature.CONCAVE)
        assert_equals(-Curvature.AFFINE, Curvature.AFFINE)

    # Tests the is_affine, is_convex, and is_concave methods
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