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
import numpy as np
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest
from cvxpy.reductions.solvers.defines \
    import INSTALLED_SOLVERS

MIP_SOLVERS = [cvx.ECOS_BB, cvx.GUROBI, cvx.MOSEK]


class TestMIPVariable(BaseTest):
    """ Unit tests for the expressions/shape module. """

    def setUp(self):
        self.x_bool = cvx.Variable(boolean=True)
        self.y_int = cvx.Variable(integer=True)
        self.A_bool = cvx.Variable((3, 2), boolean=True)
        self.B_int = cvx.Variable((2, 3), integer=True)
        # Check for all installed QP solvers
        self.solvers = [x for x in MIP_SOLVERS if x in INSTALLED_SOLVERS]

    def test_mip_consistency(self):
        """Test that MIP problems are deterministic.
        """
        data_recs = []
        for i in range(5):
            obj = cvx.Minimize(cvx.square(self.y_int - 0.2))
            p = cvx.Problem(obj, [self.A_bool == 0, self.x_bool == self.B_int])
            data_recs.append(p.get_problem_data(cvx.ECOS_BB))

        # Check that problem data and result is always the same.
        for i in range(1, 5):
            # self.assertEqual(result_recs[0], result_recs[i])
            for key in ["c", "A", "b", "G", "h",
                        "bool_vars_idx", "int_vars_idx"]:
                lh_item = data_recs[0][0][key]
                rh_item = data_recs[i][0][key]
                if key in ["A", "G"]:
                    lh_item = lh_item.todense()
                    rh_item = rh_item.todense()
                self.assertItemsAlmostEqual(lh_item, rh_item)

    # def test_mip_print(self):
    #     """Test to string methods for Bool/Int vars.
    #     """
    #     self.assertEqual(repr(self.x_bool), "Bool(1, 1)")
    #     self.assertEqual(repr(self.B_int), "Int(2, 3)")

    def test_all_solvers(self):
        for solver in self.solvers:
            self.bool_prob(solver)
            self.int_prob(solver)
            self.bool_socp(solver)
            self.int_socp(solver)

    def bool_prob(self, solver):
        # Bool in objective.
        obj = cvx.Minimize(cvx.square(self.x_bool - 0.2))
        p = cvx.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.x_bool.value, 0)

        # Bool in constraint.
        t = cvx.Variable()
        obj = cvx.Minimize(t)
        p = cvx.Problem(obj, [cvx.square(self.x_bool) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

        self.assertAlmostEqual(self.x_bool.value, 0, places=4)

        # Matrix Bool in objective.
        C = np.array([[0, 1, 0], [1, 1, 1]]).T
        obj = cvx.Minimize(cvx.sum_squares(self.A_bool - C))
        p = cvx.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

        # Matrix Bool in constraint.
        t = cvx.Variable()
        obj = cvx.Minimize(t)
        p = cvx.Problem(obj, [cvx.sum_squares(self.A_bool - C) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0)

        self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)

    def int_prob(self, solver):
        # Int in objective.
        obj = cvx.Minimize(cvx.square(self.y_int - 0.2))
        p = cvx.Problem(obj, [])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.y_int.value, 0)

        # Infeasible integer problem
        obj = cvx.Minimize(0)
        p = cvx.Problem(obj, [self.y_int == 0.5])
        result = p.solve(solver=solver)
        self.assertEqual(p.status in s.INF_OR_UNB, True)

    def int_socp(self, solver):
        # Int in objective.
        t = cvx.Variable()
        obj = cvx.Minimize(t)
        p = cvx.Problem(obj, [cvx.square(self.y_int - 0.2) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.y_int.value, 0)

    def bool_socp(self, solver):
        # Int in objective.
        t = cvx.Variable()
        obj = cvx.Minimize(t)
        p = cvx.Problem(obj, [cvx.square(self.x_bool - 0.2) <= t])
        result = p.solve(solver=solver)
        self.assertAlmostEqual(result, 0.04)

        self.assertAlmostEqual(self.x_bool.value, 0)
