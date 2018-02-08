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

import cvxpy as cvx
from cvxpy.expressions.variable import Variable
import numpy as np
from cvxpy.tests.base_test import BaseTest
import unittest


class TestSemidefiniteVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """

    def setUp(self):
        self.X = Variable((2, 2), PSD=True)
        self.Y = Variable((2, 2))
        self.F = np.matrix([[1, 0], [0, -1]])

    def test_symm(self):
        """Test that results are symmetric.
        """
        M = Variable((3, 3), PSD=True)
        C1 = np.array([[0, 0, 1/2], [0, 0, 0], [1/2, 0, 1]])
        C2 = np.array([[0, 0, 0], [0, 0, 1/2], [0, 1/2, 1]])
        x1 = Variable((3,3), PSD=True)
        x2 = Variable((3,3), PSD=True)
        constraints = [M + C1 == x1]
        constraints += [M + C2 == x2]
        objective = cvx.Minimize(cvx.trace(M))
        prob = cvx.Problem(objective, constraints)
        opt_val = prob.solve()
        assert (M.value == M.T.value).all()

    def test_sdp_problem(self):
        # PSD in objective.
        obj = cvx.Minimize(cvx.sum(cvx.square(self.X - self.F)))
        p = cvx.Problem(obj, [])
        result = p.solve()
        self.assertAlmostEqual(result, 1, places=4)

        self.assertAlmostEqual(self.X.value[0, 0], 1, places=3)
        self.assertAlmostEqual(self.X.value[0, 1], 0)
        self.assertAlmostEqual(self.X.value[1, 0], 0)
        self.assertAlmostEqual(self.X.value[1, 1], 0)

        # PSD in constraint.
        # ECHU: note to self, apparently this is a source of redundancy
        obj = cvx.Minimize(cvx.sum(cvx.square(self.Y - self.F)))
        p = cvx.Problem(obj, [self.Y == Variable((2, 2), PSD=True)])
        result = p.solve()
        self.assertAlmostEqual(result, 1, places=2)

        self.assertAlmostEqual(self.Y.value[0, 0], 1, places=3)
        self.assertAlmostEqual(self.Y.value[0, 1], 0)
        self.assertAlmostEqual(self.Y.value[1, 0], 0)
        self.assertAlmostEqual(self.Y.value[1, 1], 0, places=3)

        # Index into semidef.
        obj = cvx.Minimize(cvx.square(self.X[0, 0] - 1) +
                       cvx.square(self.X[1, 0] - 2) +
                       #square(self.X[0,1] - 3) +
                       cvx.square(self.X[1, 1] - 4))
        p = cvx.Problem(obj, [])
        result = p.solve()
        print(self.X.value)
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.X.value[0, 0], 1, places=2)
        self.assertAlmostEqual(self.X.value[0, 1], 2, places=2)
        self.assertAlmostEqual(self.X.value[1, 0], 2, places=2)
        self.assertAlmostEqual(self.X.value[1, 1], 4, places=3)
