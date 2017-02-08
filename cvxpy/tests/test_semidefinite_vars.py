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

from cvxpy import *
from cvxpy.expressions.variables import Semidef
from cvxpy.expressions.variables.semidef_var import Semidef as semidefinite
from cvxpy.expressions.variables.semidef_var import SemidefUpperTri
import numpy as np
from cvxpy.tests.base_test import BaseTest
import unittest


class TestSemidefiniteVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """

    def setUp(self):
        self.X = Semidef(2)
        self.Y = Variable(2, 2)
        self.F = np.matrix([[1, 0], [0, -1]])

    def test_sdp_print(self):
        """Test to string methods for SDP vars.
        """
        self.assertEqual(repr(SemidefUpperTri(2)), "SemidefUpperTri(2)")

    def test_sdp_problem(self):
        # SDP in objective.
        obj = Minimize(sum_entries(square(self.X - self.F)))
        p = Problem(obj, [])
        result = p.solve()
        self.assertAlmostEqual(result, 1, places=4)

        self.assertAlmostEqual(self.X.value[0, 0], 1, places=3)
        self.assertAlmostEqual(self.X.value[0, 1], 0)
        self.assertAlmostEqual(self.X.value[1, 0], 0)
        self.assertAlmostEqual(self.X.value[1, 1], 0)

        # SDP in constraint.
        # ECHU: note to self, apparently this is a source of redundancy
        obj = Minimize(sum_entries(square(self.Y - self.F)))
        p = Problem(obj, [self.Y == Semidef(2)])
        result = p.solve()
        self.assertAlmostEqual(result, 1, places=2)

        self.assertAlmostEqual(self.Y.value[0, 0], 1, places=3)
        self.assertAlmostEqual(self.Y.value[0, 1], 0)
        self.assertAlmostEqual(self.Y.value[1, 0], 0)
        self.assertAlmostEqual(self.Y.value[1, 1], 0, places=3)

        # Index into semidef.
        obj = Minimize(square(self.X[0, 0] - 1) +
                       square(self.X[1, 0] - 2) +
                       #square(self.X[0,1] - 3) +
                       square(self.X[1, 1] - 4))
        p = Problem(obj, [])
        result = p.solve()
        print(self.X.value)
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.X.value[0, 0], 1, places=2)
        self.assertAlmostEqual(self.X.value[0, 1], 2, places=2)
        self.assertAlmostEqual(self.X.value[1, 0], 2, places=2)
        self.assertAlmostEqual(self.X.value[1, 1], 4, places=3)

    def test_legacy(self):
        """Test that the legacy name semidefinite works.
        """
        X = semidefinite(2)
        obj = Minimize(sum_entries(square(X - self.F)))
        p = Problem(obj, [])
        result = p.solve()
        self.assertAlmostEqual(result, 1, places=4)
