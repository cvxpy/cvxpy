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

import cvxpy as cvx
import cvxpy.settings as s
from nose.tools import *

class TestMonotonicity(object):
    """ Unit tests for the utilities/monotonicity class. """
    # Test application of DCP composition rules to determine curvature.
    def test_dcp_curvature(self):
      expr = 1 + cvx.exp(cvx.Variable())
      assert_equals(expr.curvature, s.CONVEX)

      expr = cvx.Parameter()*cvx.NonNegative()
      assert_equals(expr.curvature, s.AFFINE)

      f = lambda x: x**2 + x**0.5
      expr = f(cvx.Constant(2))
      assert_equals(expr.curvature, s.CONSTANT)

      expr = cvx.exp(cvx.Variable())**2
      assert_equals(expr.curvature, s.CONVEX)

      expr = 1 - cvx.sqrt(cvx.Variable())
      assert_equals(expr.curvature, s.CONVEX)

      expr = cvx.log( cvx.sqrt(cvx.Variable()) )
      assert_equals(expr.curvature, s.CONCAVE)

      expr = -( cvx.exp(cvx.Variable()) )**2
      assert_equals(expr.curvature, s.CONCAVE)

      expr = cvx.log( cvx.exp(cvx.Variable()) )
      assert_equals(expr.is_dcp(), False)

      expr = cvx.entr( cvx.NonNegative() )
      assert_equals(expr.curvature, s.CONCAVE)

      expr = ( (cvx.Variable()**2)**0.5 )**0
      assert_equals(expr.curvature, s.CONSTANT)

    # Test DCP composition rules with signed monotonicity.
    def test_signed_curvature(self):
        # Convex argument.
        expr = cvx.abs(1 + cvx.exp(cvx.Variable()) )
        assert_equals(expr.curvature, s.CONVEX)

        expr = cvx.abs( -cvx.entr(cvx.Variable()) )
        assert_equals(expr.curvature, s.UNKNOWN)

        expr = cvx.abs( -cvx.log(cvx.Variable()) )
        assert_equals(expr.curvature, s.UNKNOWN)

        # Concave argument.
        expr = cvx.abs( cvx.log(cvx.Variable()) )
        assert_equals(expr.curvature, s.UNKNOWN)

        expr = cvx.abs( -cvx.square(cvx.Variable()) )
        assert_equals(expr.curvature, s.CONVEX)

        expr = cvx.abs( cvx.entr(cvx.Variable()) )
        assert_equals(expr.curvature, s.UNKNOWN)

        # Affine argument.
        expr = cvx.abs( cvx.NonNegative() )
        assert_equals(expr.curvature, s.CONVEX)

        expr = cvx.abs( -cvx.NonNegative() )
        assert_equals(expr.curvature, s.CONVEX)

        expr = cvx.abs( cvx.Variable() )
        assert_equals(expr.curvature, s.CONVEX)
