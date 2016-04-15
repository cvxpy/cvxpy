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

import cvxpy
import cvxpy.settings as s
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative, Bool, Int
from cvxpy.expressions.constants import Parameter
import cvxpy.utilities as u
import numpy as np
import unittest
from cvxpy import Problem, Minimize
from cvxpy.tests.base_test import BaseTest

class TestDomain(BaseTest):
    """ Unit tests for the domain module. """
    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    def test_log(self):
        """Test domain for log.
        """
        dom = log(self.a).domain
        Problem(Minimize(self.a), dom).solve()
        self.assertAlmostEquals(self.a.value, 0)

    def test_power(self):
        """Test domain for power.
        """
        dom = sqrt(self.a).domain
        Problem(Minimize(self.a), dom).solve()
        self.assertAlmostEquals(self.a.value, 0)

        dom = square(self.a).domain
        Problem(Minimize(self.a), dom + [self.a >= -100]).solve()
        self.assertAlmostEquals(self.a.value, -100)

        dom = ((self.a)**-1).domain
        Problem(Minimize(self.a), dom + [self.a >= -100]).solve()
        self.assertAlmostEquals(self.a.value, 0)

        dom = ((self.a)**3).domain
        Problem(Minimize(self.a), dom + [self.a >= -100]).solve()
        self.assertAlmostEquals(self.a.value, 0)

    def test_log_det(self):
        """Test domain for log_det.
        """
        dom = log_det(self.A + np.eye(2)).domain
        prob = Problem(Minimize(sum_entries(diag(self.A))), dom)
        prob.solve(solver=cvxpy.SCS)
        self.assertAlmostEquals(prob.value, -2, places=3)
