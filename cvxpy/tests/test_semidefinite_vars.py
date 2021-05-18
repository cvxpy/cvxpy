"""
Copyright 2013 Steven Diamond, Eric Chu

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

import cvxpy as cvx
from cvxpy.expressions.variable import Variable
import numpy as np
from cvxpy.tests.base_test import BaseTest


class TestSemidefiniteVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """

    def setUp(self) -> None:
        self.X = Variable((2, 2), PSD=True)
        self.Y = Variable((2, 2))
        self.F = np.array([[1, 0], [0, -1]])

    def test_symm(self) -> None:
        """Test that results are symmetric.
        """
        M = Variable((3, 3), PSD=True)
        C1 = np.array([[0, 0, 1/2], [0, 0, 0], [1/2, 0, 1]])
        C2 = np.array([[0, 0, 0], [0, 0, 1/2], [0, 1/2, 1]])
        x1 = Variable((3, 3), PSD=True)
        x2 = Variable((3, 3), PSD=True)
        constraints = [M + C1 == x1]
        constraints += [M + C2 == x2]
        objective = cvx.Minimize(cvx.trace(M))
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        assert (M.value == M.T.value).all()

    def test_sdp_problem(self) -> None:
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
                           # square(self.X[0,1] - 3) +
                           cvx.square(self.X[1, 1] - 4))
        p = cvx.Problem(obj, [])
        result = p.solve()
        print(self.X.value)
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.X.value[0, 0], 1, places=2)
        self.assertAlmostEqual(self.X.value[0, 1], 2, places=2)
        self.assertAlmostEqual(self.X.value[1, 0], 2, places=2)
        self.assertAlmostEqual(self.X.value[1, 1], 4, places=3)
