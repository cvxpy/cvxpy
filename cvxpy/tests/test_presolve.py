"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy.problems import utils
from cvxopt import matrix
from base_test import BaseTest
import unittest


class TestPresolve(BaseTest):
    """ Unit tests for the expressions/shape module. """
    def setUp(self):
        self.A = matrix([[1,-1], [0,0]])
        self.b = matrix([3,-3])
    
    def test_rows(self):
        i = 0
        for r in utils.mat_rows(self.A):
            self.assertItemsAlmostEqual(r, self.A[i,:])
            i += 1
        
    def test_normalize(self):
        H = self.A
        g = self.b
        H, g, _ = utils.normalize_data(H,g)
        print H, g
        self.assertItemsAlmostEqual(H[0], self.A[0,:])
        self.assertItemsAlmostEqual(H[1], -self.A[0,:])
        self.assertAlmostEqual(g[0], -g[1])
        
        H = matrix([[1,-0.5],[0, 0], [-0.5, 0.25]])
        g = matrix([1, -0.5])
        H, g, _ = utils.normalize_data(H,g)
        print H, g
        self.assertItemsAlmostEqual(H[0], [1,0,-0.5])
        self.assertItemsAlmostEqual(H[1], [-1,0,0.5])
        self.assertAlmostEqual(g[0], -g[1])

    
    def test_presolve(self):
        H = self.A
        g = self.b
        
        H, g = utils.remove_redundant_rows(H, g)
        assert H.size[0] == 1
        self.assertItemsAlmostEqual(H[0,:], self.A[0,:])
        self.assertAlmostEqual(g[0], 3)
        
        H = matrix([[1,-0.5],[0, 0], [-0.5, 0.25]])
        g = matrix([1, -0.5])
        
        H, g = utils.remove_redundant_rows(H,g)
        assert H.size[0] == 1
        self.assertItemsAlmostEqual(H[0,:], [1,-0.5])
        self.assertAlmostEqual(g[0], 1)
        
        H, g = matrix([]), matrix([])
        Hout, gout = utils.remove_redundant_rows(H,g)
        assert Hout is H
        assert gout is g

   