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
import unittest

import numpy as np

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest


@unittest.skipUnless(len(MIP_SOLVERS) > 0, 'No mixed-integer solver is installed.')
class TestMIPVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """

    def setUp(self) -> None:
        self.x_bool = cp.Variable(boolean=True)
        self.y_int = cp.Variable(integer=True)
        self.A_bool = cp.Variable((3, 2), boolean=True)
        self.B_int = cp.Variable((2, 3), integer=True)
        # Check for all installed QP solvers
        self.solvers = MIP_SOLVERS

    def test_all_solvers(self) -> None:
        for solver in self.solvers:
            self.bool_prob(solver)
            if solver != cp.SCIPY:
                self.int_prob(solver)  # issue #1938
            if solver in [cp.CPLEX, cp.GUROBI, cp.MOSEK, cp.XPRESS]:
                if solver != cp.XPRESS:  # issue #1815
                    self.bool_socp(solver)
                self.int_socp(solver)

    def bool_prob(self, solver) -> None:
        # Bool in objective.
        obj = cp.Minimize(cp.abs(self.x_bool - 0.2))
        p = cp.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.2)

        self.assertAlmostEqual(self.x_bool.value, 0)

        # Bool in constraint.
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.abs(self.x_bool) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.x_bool.value, 0, places=4)

        # Matrix Bool in objective.
        C = np.array([[0, 1, 0], [1, 1, 1]]).T
        obj = cp.Minimize(cp.sum(cp.abs(self.A_bool - C)))
        p = cp.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

        # Matrix Bool in constraint.
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.sum(cp.abs(self.A_bool - C)) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

    def int_prob(self, solver) -> None:
        # Int in objective.
        obj = cp.Minimize(cp.abs(self.y_int - 0.2))
        p = cp.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.2)

        self.assertAlmostEqual(self.y_int.value, 0)

        # Infeasible integer problem
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [self.y_int == 0.5, t >= 0])
        result = p.solve(solver=solver)
        self.assertEqual(p.status in s.INF_OR_UNB, True)

    def int_socp(self, solver) -> None:
        # Int in objective.
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.square(self.y_int - 0.2) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.y_int.value, 0)

    def bool_socp(self, solver) -> None:
        # Bool in objective.
        t = cp.Variable()
        obj = cp.Minimize(t)
        p = cp.Problem(obj, [cp.square(self.x_bool - 0.2) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.x_bool.value, 0)
