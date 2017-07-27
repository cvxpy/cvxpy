"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy as cvx
import cvxpy.settings as s
from nose.tools import *
from cvxpy.tests.base_test import BaseTest


class TestMonotonicity(BaseTest):
    """ Unit tests for the utilities/monotonicity class. """
    # Test application of DCP composition rules to determine curvature.

    def test_dcp_curvature(self):
        expr = 1 + cvx.exp(cvx.Variable())
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cvx.Parameter()*cvx.NonNegative()
        self.assertEqual(expr.curvature, s.AFFINE)

        f = lambda x: x**2 + x**0.5
        expr = f(cvx.Constant(2))
        self.assertEqual(expr.curvature, s.CONSTANT)

        expr = cvx.exp(cvx.Variable())**2
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = 1 - cvx.sqrt(cvx.Variable())
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cvx.log(cvx.sqrt(cvx.Variable()))
        self.assertEqual(expr.curvature, s.CONCAVE)

        expr = -(cvx.exp(cvx.Variable()))**2
        self.assertEqual(expr.curvature, s.CONCAVE)

        expr = cvx.log(cvx.exp(cvx.Variable()))
        self.assertEqual(expr.is_dcp(), False)

        expr = cvx.entr(cvx.NonNegative())
        self.assertEqual(expr.curvature, s.CONCAVE)

        expr = ((cvx.Variable()**2)**0.5)**0
        self.assertEqual(expr.curvature, s.CONSTANT)

    # Test DCP composition rules with signed monotonicity.
    def test_signed_curvature(self):
        # Convex argument.
        expr = cvx.abs(1 + cvx.exp(cvx.Variable()))
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cvx.abs(-cvx.entr(cvx.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        expr = cvx.abs(-cvx.log(cvx.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        # Concave argument.
        expr = cvx.abs(cvx.log(cvx.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        expr = cvx.abs(-cvx.square(cvx.Variable()))
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cvx.abs(cvx.entr(cvx.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        # Affine argument.
        expr = cvx.abs(cvx.NonNegative())
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cvx.abs(-cvx.NonNegative())
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cvx.abs(cvx.Variable())
        self.assertEqual(expr.curvature, s.CONVEX)
