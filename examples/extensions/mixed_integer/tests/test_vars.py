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
        p = Problem(Minimize(sum(1-x) + sum(x)), [x == BoolVar(5,4)])
        result = p.solve(method="admm", solver="cvxopt")
        self.assertAlmostEqual(result, 20)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)

        x = Variable()
        p = Problem(Minimize(sum(1-x) + sum(x)), [x == BoolVar(5,4)[0,0]])
        result = p.solve(method="admm", solver="cvxopt")
        self.assertAlmostEqual(result, 1)
        self.assertAlmostEqual(x.value*(1-x.value), 0)

    # Test choose variable.
    def test_choose(self):
        x = Variable(5,4)
        p = Problem(Minimize(sum(1-x) + sum(x)), 
                    [x == SparseBoolVar(5,4,nonzeros=4)])
        result = p.solve(method="admm", solver="cvxopt")
        self.assertAlmostEqual(result, 20)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(sum(x.value), 4)

    # Test card variable.
    def test_card(self):
        x = SparseVar(5,nonzeros=3)
        p = Problem(Maximize(sum(x)),
            [x <= 1, x >= 0])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 3)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(sum(x.value), 3)

        #should be equivalent to x == choose
        x = Variable(5,4)
        c = SparseVar(5,4,nonzeros=4)
        b = BoolVar(5,4)
        p = Problem(Minimize(sum(1-x) + sum(x)), 
            [x == c, x == b])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 20)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)

    # Test permutation variable.
    def test_permutation(self):
        x = Variable(1,5)
        c = cvxopt.matrix([1,2,3,4,5]).T
        perm = permutation(5)
        p = Problem(Minimize(sum(x)), [x == c*perm])
        result = p.solve(method="admm")
        print perm.value
        print x.value
        self.assertAlmostEqual(result, 15)
        self.assertAlmostEqual(sorted(x.value), c)