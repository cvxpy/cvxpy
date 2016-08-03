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

from cvxpy import log
from cvxpy import Constant, Variable, Parameter
import cvxpy.settings as s
from nose.tools import assert_equals


class TestCurvature(object):
    """ Unit tests for the expression/curvature class. """
    @classmethod
    def setup_class(self):
        self.cvx = Variable()**2
        self.ccv = Variable()**0.5
        self.aff = Variable()
        self.const = Constant(5)
        self.unknown_curv = log(Variable()**3)

        self.pos = Constant(1)
        self.neg = Constant(-1)
        self.zero = Constant(0)
        self.unknown_sign = Parameter()

    def test_add(self):
        assert_equals((self.const + self.cvx).curvature, self.cvx.curvature)
        assert_equals((self.unknown_curv + self.ccv).curvature, self.unknown_curv.curvature)
        assert_equals((self.cvx + self.ccv).curvature, self.unknown_curv.curvature)
        assert_equals((self.cvx + self.cvx).curvature, self.cvx.curvature)
        assert_equals((self.aff + self.ccv).curvature, self.ccv.curvature)

    def test_sub(self):
        assert_equals((self.const - self.cvx).curvature, self.ccv.curvature)
        assert_equals((self.unknown_curv - self.ccv).curvature, self.unknown_curv.curvature)
        assert_equals((self.cvx - self.ccv).curvature, self.cvx.curvature)
        assert_equals((self.cvx - self.cvx).curvature, self.unknown_curv.curvature)
        assert_equals((self.aff - self.ccv).curvature, self.cvx.curvature)

    def test_sign_mult(self):
        assert_equals((self.zero * self.cvx).curvature, self.const.curvature)
        assert_equals((self.neg*self.cvx).curvature, self.ccv.curvature)
        assert_equals((self.neg*self.ccv).curvature, self.cvx.curvature)
        assert_equals((self.neg*self.unknown_curv).curvature, self.unknown_curv.curvature)
        assert_equals((self.pos*self.aff).curvature, self.aff.curvature)
        assert_equals((self.pos*self.ccv).curvature, self.ccv.curvature)
        assert_equals((self.unknown_sign*self.const).curvature, self.const.curvature)
        assert_equals((self.unknown_sign*self.ccv).curvature, self.unknown_curv.curvature)

    def test_neg(self):
        assert_equals((-self.cvx).curvature, self.ccv.curvature)
        assert_equals((-self.aff).curvature, self.aff.curvature)

    # Tests the is_affine, is_convex, and is_concave methods
    def test_is_curvature(self):
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
