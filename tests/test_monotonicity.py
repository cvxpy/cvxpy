from cvxpy.expressions.curvature import Curvature
from cvxpy.atoms.monotonicity import Monotonicity
from nose.tools import *

class TestMonotonicity(object):
    """ Unit tests for the atoms/monotonicity class. """
    # Test application of DCP composition rules to determine curvature.
    def test_dcp_curvature(self):
        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.AFFINE, Curvature.CONVEX), Curvature.CONVEX)
        assert_equals(Monotonicity.NONMONOTONIC.dcp_curvature(Curvature.AFFINE, Curvature.AFFINE), Curvature.AFFINE)
        assert_equals(Monotonicity.DECREASING.dcp_curvature(Curvature.UNKNOWN, Curvature.CONSTANT), Curvature.CONSTANT)

        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.CONVEX, Curvature.CONVEX), Curvature.CONVEX)
        assert_equals(Monotonicity.DECREASING.dcp_curvature(Curvature.CONVEX, Curvature.CONCAVE), Curvature.CONVEX)

        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.CONCAVE, Curvature.CONCAVE), Curvature.CONCAVE)
        assert_equals(Monotonicity.DECREASING.dcp_curvature(Curvature.CONCAVE, Curvature.CONVEX), Curvature.CONCAVE)

        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.CONCAVE, Curvature.CONVEX), Curvature.UNKNOWN)
        assert_equals(Monotonicity.NONMONOTONIC.dcp_curvature(Curvature.CONCAVE, Curvature.AFFINE), Curvature.CONCAVE)

        assert_equals(Monotonicity.NONMONOTONIC.dcp_curvature(Curvature.CONSTANT, Curvature.UNKNOWN), Curvature.CONSTANT)