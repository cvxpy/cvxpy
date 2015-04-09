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

from cvxpy import *
from mixed_integer import *
import cvxopt
import numpy as np
import unittest

class TestVars(unittest.TestCase):
    """ Unit tests for the variable types. """
    def setUp(self):
        pass

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b):
        try:
            a = list(a)
            b = list(b)
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        except Exception:
            super(TestVars, self).assertAlmostEqual(a,b,places=3)

    # Test boolean variable.
    def test_boolean(self):
        x = Variable(5,4)
        p = Problem(Minimize(sum_entries(1-x) + sum_entries(x)), [x == Boolean(5,4)])
        result = p.solve(method="admm", solver=CVXOPT)
        self.assertAlmostEqual(result, 20)
        for i in xrange(x.size[0]):
            for j in xrange(x.size[1]):
                v = x.value[i, j]
                self.assertAlmostEqual(v*(1-v), 0)

        x = Variable()
        p = Problem(Minimize(sum_entries(1-x) + sum_entries(x)), [x == Boolean(5,4)[0,0]])
        result = p.solve(method="admm", solver=CVXOPT)
        self.assertAlmostEqual(result, 1)
        self.assertAlmostEqual(x.value*(1-x.value), 0)

    # Test choose variable.
    def test_choose(self):
        x = Variable(5,4)
        p = Problem(Minimize(sum_entries(1-x) + sum_entries(x)),
                    [x == Choose(5,4,k=4)])
        result = p.solve(method="admm", solver=CVXOPT)
        self.assertAlmostEqual(result, 20)
        for i in xrange(x.size[0]):
            for j in xrange(x.size[1]):
                v = x.value[i, j]
                self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(x.value.sum(), 4)

    # Test card variable.
    def test_card(self):
        x = Card(5,k=3)
        p = Problem(Maximize(sum_entries(x)),
            [x <= 1, x >= 0])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 3)
        for v in np.nditer(x.value):
            self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(x.value.sum(), 3)

        #should be equivalent to x == choose
        x = Variable(5, 4)
        c = Card(5, 4, k=4)
        b = Boolean(5, 4)
        p = Problem(Minimize(sum_entries(1-x) + sum_entries(x)),
                    [x == c, x == b])
        result = p.solve(method="admm", solver=CVXOPT)
        self.assertAlmostEqual(result, 20)
        for i in xrange(x.size[0]):
            for j in xrange(x.size[1]):
                v = x.value[i, j]
                self.assertAlmostEqual(v*(1-v), 0)

    # Test permutation variable.
    def test_permutation(self):
        x = Variable(1,5)
        c = cvxopt.matrix([1,2,3,4,5]).T
        perm = Assign(5, 5)
        p = Problem(Minimize(sum_entries(x)), [x == c*perm])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 15)
        self.assertAlmostEqual(sorted(np.nditer(x.value)), c)
