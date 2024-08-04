"""
Copyright 2013 Steven Diamond

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

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest


class TestMonotonicity(BaseTest):
    """ Unit tests for the utilities/monotonicity class. """
    # Test application of DCP composition rules to determine curvature.

    def test_dcp_curvature(self) -> None:
        expr = 1 + cp.exp(cp.Variable())
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cp.Parameter()*cp.Variable(nonneg=True)
        self.assertEqual(expr.curvature, s.AFFINE)

        f = lambda x: x**2 + x**0.5  # noqa E731
        expr = f(cp.Constant(2))
        self.assertEqual(expr.curvature, s.CONSTANT)

        expr = cp.exp(cp.Variable())**2
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = 1 - cp.sqrt(cp.Variable())
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cp.log(cp.sqrt(cp.Variable()))
        self.assertEqual(expr.curvature, s.CONCAVE)

        expr = -(cp.exp(cp.Variable()))**2
        self.assertEqual(expr.curvature, s.CONCAVE)

        expr = cp.log(cp.exp(cp.Variable()))
        self.assertEqual(expr.is_dcp(), False)

        expr = cp.entr(cp.Variable(nonneg=True))
        self.assertEqual(expr.curvature, s.CONCAVE)

        expr = ((cp.Variable()**2)**0.5)**0
        self.assertEqual(expr.curvature, s.CONSTANT)

    # Test DCP composition rules with signed monotonicity.
    def test_signed_curvature(self) -> None:
        # Convex argument.
        expr = cp.abs(1 + cp.exp(cp.Variable()))
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cp.abs(-cp.entr(cp.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        expr = cp.abs(-cp.log(cp.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        # Concave argument.
        expr = cp.abs(cp.log(cp.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        expr = cp.abs(-cp.square(cp.Variable()))
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cp.abs(cp.entr(cp.Variable()))
        self.assertEqual(expr.curvature, s.UNKNOWN)

        # Affine argument.
        expr = cp.abs(cp.Variable(nonneg=True))
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cp.abs(-cp.Variable(nonneg=True))
        self.assertEqual(expr.curvature, s.CONVEX)

        expr = cp.abs(cp.Variable())
        self.assertEqual(expr.curvature, s.CONVEX)
