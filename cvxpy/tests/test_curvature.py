from cvxpy.expressions.curvature import Curvature
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

    def test_mult(self):
        assert_equals(Curvature.CONSTANT * Curvature.CONVEX, Curvature.CONVEX)
        assert_equals(Curvature.CONSTANT * Curvature.AFFINE, Curvature.AFFINE)
        assert_equals(Curvature.AFFINE * Curvature.CONCAVE, Curvature.UNKNOWN)

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