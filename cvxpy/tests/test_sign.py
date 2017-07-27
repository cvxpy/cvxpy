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

from cvxpy import Constant, Variable
import cvxpy.settings as s
from nose.tools import *
from cvxpy.tests.base_test import BaseTest


class TestSign(BaseTest):
    """ Unit tests for the expression/sign class. """
    @classmethod
    def setup_class(self):
        self.pos = Constant(1)
        self.neg = Constant(-1)
        self.zero = Constant(0)
        self.unknown = Variable()

    def test_add(self):
        self.assertEqual((self.pos + self.neg).sign, self.unknown.sign)
        self.assertEqual((self.neg + self.zero).sign, self.neg.sign)
        self.assertEqual((self.pos + self.pos).sign, self.pos.sign)
        self.assertEqual((self.unknown + self.zero).sign, self.unknown.sign)

    def test_sub(self):
        self.assertEqual((self.pos - self.neg).sign, self.pos.sign)
        self.assertEqual((self.neg - self.zero).sign, self.neg.sign)
        self.assertEqual((self.pos - self.pos).sign, self.unknown.sign)

    def test_mult(self):
        self.assertEqual((self.zero * self.pos).sign, self.zero.sign)
        self.assertEqual((self.unknown * self.pos).sign, self.unknown.sign)
        self.assertEqual((self.pos * self.neg).sign, self.neg.sign)
        self.assertEqual((self.pos * self.pos).sign, self.pos.sign)
        self.assertEqual((self.pos * self.pos).sign, self.pos.sign)
        self.assertEqual((self.neg * self.neg).sign, self.pos.sign)
        self.assertEqual((self.zero * self.unknown).sign, self.zero.sign)

    def test_neg(self):
        self.assertEqual((-self.zero).sign, self.zero.sign)
        self.assertEqual((-self.pos).sign, self.neg.sign)

    # Tests the is_positive and is_negative methods.
    def test_is_sign(self):
        assert self.pos.is_positive()
        assert not self.neg.is_positive()
        assert not self.unknown.is_positive()
        assert self.zero.is_positive()

        assert not self.pos.is_negative()
        assert self.neg.is_negative()
        assert not self.unknown.is_negative()
        assert self.zero.is_negative()

        assert self.zero.is_zero()
        assert not self.neg.is_zero()

        assert not (self.unknown.is_positive() or self.unknown.is_negative())
