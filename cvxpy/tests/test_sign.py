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

from cvxpy import Constant, Variable
import cvxpy.settings as s
from nose.tools import *

class TestSign(object):
  """ Unit tests for the expression/sign class. """
  @classmethod
  def setup_class(self):
      self.pos = Constant(1)
      self.neg = Constant(-1)
      self.zero = Constant(0)
      self.unknown = Variable()

  def test_add(self):
      assert_equals( (self.pos + self.neg).sign, self.unknown.sign)
      assert_equals( (self.neg + self.zero).sign, self.neg.sign)
      assert_equals( (self.pos + self.pos).sign, self.pos.sign)
      assert_equals( (self.unknown + self.zero).sign, self.unknown.sign)

  def test_sub(self):
      assert_equals( (self.pos - self.neg).sign, self.pos.sign)
      assert_equals( (self.neg - self.zero).sign, self.neg.sign)
      assert_equals( (self.pos - self.pos).sign, self.unknown.sign)

  def test_mult(self):
      assert_equals( (self.zero * self.pos).sign, self.zero.sign)
      assert_equals( (self.unknown * self.pos).sign, self.unknown.sign)
      assert_equals( (self.pos * self.neg).sign, self.neg.sign)
      assert_equals( (self.pos * self.pos).sign, self.pos.sign)
      assert_equals( (self.pos * self.pos).sign, self.pos.sign)
      assert_equals( (self.neg * self.neg).sign, self.pos.sign)
      assert_equals( (self.zero * self.unknown).sign, self.zero.sign)

  def test_neg(self):
      assert_equals( (-self.zero).sign, self.zero.sign)
      assert_equals( (-self.pos).sign, self.neg.sign)

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
