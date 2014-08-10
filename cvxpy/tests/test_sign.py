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

from cvxpy.utilities import Sign
from nose.tools import *

class TestSign(object):
  """ Unit tests for the expression/sign class. """
  @classmethod
  def setup_class(self):
      pass

  def test_add(self):
      assert_equals(Sign.POSITIVE + Sign.NEGATIVE, Sign.UNKNOWN)
      assert_equals(Sign.NEGATIVE + Sign.ZERO, Sign.NEGATIVE)
      assert_equals(Sign.POSITIVE + Sign.POSITIVE, Sign.POSITIVE)
      assert_equals(Sign.UNKNOWN + Sign.ZERO, Sign.UNKNOWN)

  def test_sub(self):
      assert_equals(Sign.POSITIVE - Sign.NEGATIVE, Sign.POSITIVE)
      assert_equals(Sign.NEGATIVE - Sign.ZERO, Sign.NEGATIVE)
      assert_equals(Sign.POSITIVE - Sign.POSITIVE, Sign.UNKNOWN)

  def test_mult(self):
      assert_equals(Sign.ZERO * Sign.POSITIVE, Sign.ZERO)
      assert_equals(Sign.UNKNOWN * Sign.POSITIVE, Sign.UNKNOWN)
      assert_equals(Sign.POSITIVE * Sign.NEGATIVE, Sign.NEGATIVE)
      assert_equals(Sign.POSITIVE * Sign.POSITIVE, Sign.POSITIVE)
      assert_equals(Sign.POSITIVE * Sign("positive"), Sign.POSITIVE)
      assert_equals(Sign.NEGATIVE * Sign.NEGATIVE, Sign.POSITIVE)
      assert_equals(Sign.ZERO * Sign.UNKNOWN, Sign.ZERO)

  def test_neg(self):
      assert_equals(-Sign.ZERO, Sign.ZERO)
      assert_equals(-Sign.POSITIVE, Sign.NEGATIVE)

  # Tests the is_positive and is_negative methods.
  def test_is_sign(self):
      assert Sign.POSITIVE.is_positive()
      assert not Sign.NEGATIVE.is_positive()
      assert not Sign.UNKNOWN.is_positive()
      assert Sign.ZERO.is_positive()

      assert not Sign.POSITIVE.is_negative()
      assert Sign.NEGATIVE.is_negative()
      assert not Sign.UNKNOWN.is_negative()
      assert Sign.ZERO.is_negative()

      assert Sign.ZERO.is_zero()
      assert not Sign.NEGATIVE.is_zero()

      assert Sign.UNKNOWN.is_unknown()
