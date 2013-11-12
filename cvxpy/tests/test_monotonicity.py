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

from cvxpy.utilities import Curvature, Sign
import cvxpy.utilities.monotonicity as m
from nose.tools import *

class TestMonotonicity(object):
    """ Unit tests for the utilities/monotonicity class. """
    # Test application of DCP composition rules to determine curvature.
    def test_dcp_curvature(self):
        assert_equals(m.dcp_curvature(m.INCREASING,
                                      Curvature.AFFINE,
                                                 Sign.POSITIVE,
                                                 Curvature.CONVEX),
                      Curvature.CONVEX)
        assert_equals(m.dcp_curvature(m.NONMONOTONIC, Curvature.AFFINE,
                                                   Sign.POSITIVE,
                                                   Curvature.AFFINE),
                    Curvature.AFFINE)
        assert_equals(m.dcp_curvature(m.DECREASING, Curvature.UNKNOWN,
                                                 Sign.POSITIVE,
                                                 Curvature.CONSTANT),
                      Curvature.CONSTANT)

        assert_equals(m.dcp_curvature(m.INCREASING, Curvature.CONVEX,
                                                            Sign.POSITIVE,
                                                            Curvature.CONVEX),
                       Curvature.CONVEX)
        assert_equals(m.dcp_curvature(m.DECREASING, Curvature.CONVEX,
                                                            Sign.POSITIVE,
                                                            Curvature.CONCAVE),
                       Curvature.CONVEX)

        assert_equals(m.dcp_curvature(m.INCREASING, Curvature.CONCAVE,
                                                            Sign.POSITIVE,
                                                            Curvature.CONCAVE),
                      Curvature.CONCAVE)
        assert_equals(m.dcp_curvature(m.DECREASING, Curvature.CONCAVE,
                                                            Sign.POSITIVE,
                                                            Curvature.CONVEX),
                      Curvature.CONCAVE)

        assert_equals(m.dcp_curvature(m.INCREASING, Curvature.CONCAVE,
                                                            Sign.POSITIVE,
                                                            Curvature.CONVEX),
                      Curvature.UNKNOWN)
        assert_equals(m.dcp_curvature(m.NONMONOTONIC, Curvature.CONCAVE,
                                                              Sign.POSITIVE,
                                                              Curvature.AFFINE),
                      Curvature.CONCAVE)

        assert_equals(m.dcp_curvature(m.NONMONOTONIC, Curvature.CONSTANT,
                                                              Sign.POSITIVE,
                                                              Curvature.UNKNOWN),
                      Curvature.UNKNOWN)

    # Test DCP composition rules with signed monotonicity.
    def test_signed_curvature(self):
        # Convex argument.
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.POSITIVE,
                                                        Curvature.CONVEX),
                      Curvature.CONVEX)
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.NEGATIVE,
                                                        Curvature.CONVEX),
                      Curvature.UNKNOWN)
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.UNKNOWN,
                                                        Curvature.CONVEX),
                      Curvature.UNKNOWN)
        # Concave argument.
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.POSITIVE,
                                                        Curvature.CONCAVE),
                      Curvature.UNKNOWN)
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.NEGATIVE,
                                                        Curvature.CONCAVE),
                      Curvature.CONVEX)
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.UNKNOWN,
                                                        Curvature.CONCAVE),
                      Curvature.UNKNOWN)
        # Affine argument.
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.POSITIVE,
                                                        Curvature.AFFINE),
                      Curvature.CONVEX)
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.NEGATIVE,
                                                        Curvature.AFFINE),
                      Curvature.CONVEX)
        assert_equals(m.dcp_curvature(m.SIGNED, Curvature.CONVEX,
                                                        Sign.UNKNOWN,
                                                        Curvature.AFFINE),
                      Curvature.CONVEX)