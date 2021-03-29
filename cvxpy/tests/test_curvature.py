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

import unittest
from cvxpy import log
from cvxpy import Constant, Variable
from cvxpy.settings import UNKNOWN, QUASILINEAR


class TestCurvature(unittest.TestCase):
    """ Unit tests for the expression/curvature class. """

    def setUp(self) -> None:
        self.cvx = Variable()**2
        self.ccv = Variable()**0.5
        self.aff = Variable()
        self.const = Constant(5)
        self.unknown_curv = log(Variable()**3)

        self.pos = Constant(1)
        self.neg = Constant(-1)
        self.zero = Constant(0)
        self.unknown_sign = self.pos + self.neg

    def test_add(self) -> None:
        self.assertEqual((self.const + self.cvx).curvature, self.cvx.curvature)
        self.assertEqual((self.unknown_curv + self.ccv).curvature, UNKNOWN)
        self.assertEqual((self.cvx + self.ccv).curvature, UNKNOWN)
        self.assertEqual((self.cvx + self.cvx).curvature, self.cvx.curvature)
        self.assertEqual((self.aff + self.ccv).curvature, self.ccv.curvature)

    def test_sub(self) -> None:
        self.assertEqual((self.const - self.cvx).curvature, self.ccv.curvature)
        self.assertEqual((self.unknown_curv - self.ccv).curvature, UNKNOWN)
        self.assertEqual((self.cvx - self.ccv).curvature, self.cvx.curvature)
        self.assertEqual((self.cvx - self.cvx).curvature, UNKNOWN)
        self.assertEqual((self.aff - self.ccv).curvature, self.cvx.curvature)

    def test_sign_mult(self) -> None:
        self.assertEqual((self.zero * self.cvx).curvature, self.aff.curvature)
        self.assertEqual((self.neg*self.cvx).curvature, self.ccv.curvature)
        self.assertEqual((self.neg*self.ccv).curvature, self.cvx.curvature)
        self.assertEqual((self.neg*self.unknown_curv).curvature, QUASILINEAR)
        self.assertEqual((self.pos*self.aff).curvature, self.aff.curvature)
        self.assertEqual((self.pos*self.ccv).curvature, self.ccv.curvature)
        self.assertEqual((self.unknown_sign*self.const).curvature, self.const.curvature)
        self.assertEqual((self.unknown_sign*self.ccv).curvature, UNKNOWN)

    def test_neg(self) -> None:
        self.assertEqual((-self.cvx).curvature, self.ccv.curvature)
        self.assertEqual((-self.aff).curvature, self.aff.curvature)

    # Tests the is_affine, is_convex, and is_concave methods
    def test_is_curvature(self) -> None:
        assert self.const.is_affine()
        assert self.aff.is_affine()
        assert not self.cvx.is_affine()
        assert not self.ccv.is_affine()
        assert not self.unknown_curv.is_affine()

        assert self.const.is_convex()
        assert self.aff.is_convex()
        assert self.cvx.is_convex()
        assert not self.ccv.is_convex()
        assert not self.unknown_curv.is_convex()

        assert self.const.is_concave()
        assert self.aff.is_concave()
        assert not self.cvx.is_concave()
        assert self.ccv.is_concave()
        assert not self.unknown_curv.is_concave()
