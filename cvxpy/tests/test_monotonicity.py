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

from cvxpy.utilities import Monotonicity, Curvature
from nose.tools import *

class TestMonotonicity(object):
    """ Unit tests for the utilities/monotonicity class. """
    # Test application of DCP composition rules to determine curvature.
    def test_dcp_curvature(self):
        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.AFFINE, Curvature.CONVEX), Curvature.CONVEX)
        assert_equals(Monotonicity.NONMONOTONIC.dcp_curvature(Curvature.AFFINE, Curvature.AFFINE), Curvature.AFFINE)
        assert_equals(Monotonicity.DECREASING.dcp_curvature(Curvature.UNKNOWN, Curvature.CONSTANT), Curvature.UNKNOWN)

        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.CONVEX, Curvature.CONVEX), Curvature.CONVEX)
        assert_equals(Monotonicity.DECREASING.dcp_curvature(Curvature.CONVEX, Curvature.CONCAVE), Curvature.CONVEX)

        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.CONCAVE, Curvature.CONCAVE), Curvature.CONCAVE)
        assert_equals(Monotonicity.DECREASING.dcp_curvature(Curvature.CONCAVE, Curvature.CONVEX), Curvature.CONCAVE)

        assert_equals(Monotonicity.INCREASING.dcp_curvature(Curvature.CONCAVE, Curvature.CONVEX), Curvature.UNKNOWN)
        assert_equals(Monotonicity.NONMONOTONIC.dcp_curvature(Curvature.CONCAVE, Curvature.AFFINE), Curvature.CONCAVE)

        assert_equals(Monotonicity.NONMONOTONIC.dcp_curvature(Curvature.CONSTANT, Curvature.UNKNOWN), Curvature.UNKNOWN)