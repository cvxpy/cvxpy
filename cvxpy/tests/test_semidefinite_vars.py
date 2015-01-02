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

from cvxpy import *
from cvxpy.expressions.variables import Semidef
from cvxpy.expressions.variables.semidef_var import Semidef as semidefinite
from cvxpy.expressions.variables.semidef_var import SemidefUpperTri
from cvxopt import matrix
import numpy as np
from cvxpy.tests.base_test import BaseTest
import unittest

class TestSemidefiniteVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """
    def setUp(self):
        self.X = Semidef(2)
        self.Y = Variable(2,2)
        self.F = matrix([[1,0],[0,-1]], tc='d')

    def test_sdp_print(self):
        """Test to string methods for SDP vars.
        """
        self.assertEqual(repr(SemidefUpperTri(2)), "SemidefUpperTri(2)")

    def test_sdp_problem(self):
        # SDP in objective.
        obj = Minimize(sum_entries(square(self.X - self.F)))
        p = Problem(obj,[])
        result = p.solve()
        self.assertAlmostEqual(result, 1)

        self.assertAlmostEqual(self.X.value[0,0], 1, places=3)
        self.assertAlmostEqual(self.X.value[0,1], 0)
        self.assertAlmostEqual(self.X.value[1,0], 0)
        self.assertAlmostEqual(self.X.value[1,1], 0)

        # SDP in constraint.
        # ECHU: note to self, apparently this is a source of redundancy
        obj = Minimize(sum_entries(square(self.Y - self.F)))
        p = Problem(obj, [self.Y == Semidef(2)])
        result = p.solve()
        self.assertAlmostEqual(result, 1)

        self.assertAlmostEqual(self.Y.value[0,0], 1, places=3)
        self.assertAlmostEqual(self.Y.value[0,1], 0)
        self.assertAlmostEqual(self.Y.value[1,0], 0)
        self.assertAlmostEqual(self.Y.value[1,1], 0)

        # Index into semidef.
        obj = Minimize(square(self.X[0,0] - 1) +
                       square(self.X[1,0] - 2) +
                       #square(self.X[0,1] - 3) +
                       square(self.X[1,1] - 4))
        p = Problem(obj,[])
        result = p.solve()
        print(self.X.value)
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.X.value[0,0], 1, places=3)
        self.assertAlmostEqual(self.X.value[0,1], 2, places=3)
        self.assertAlmostEqual(self.X.value[1,0], 2, places=3)
        self.assertAlmostEqual(self.X.value[1,1], 4, places=4)

    def test_legacy(self):
        """Test that the legacy name semidefinite works.
        """
        X = semidefinite(2)
        obj = Minimize(sum_entries(square(X - self.F)))
        p = Problem(obj,[])
        result = p.solve()
        self.assertAlmostEqual(result, 1)
