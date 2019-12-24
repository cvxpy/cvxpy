"""
Copyright, the CVXPY authors

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
import math

import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.tests.base_test import BaseTest

import numpy as np


class TestParamConeProg(BaseTest):
    # Overridden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestParamConeProg, self).assertItemsAlmostEqual(a, b, places=places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestParamConeProg, self).assertAlmostEqual(a, b, places=places)

    def test_log_problem(self):
        # Log in objective.
        x = cp.Variable(2)
        var_dict = {x.id: x}
        obj = cp.Maximize(cp.sum(cp.log(x)))
        constr = [x <= [1, math.e]]
        problem = cp.Problem(obj, constr)
        data, _, _ = problem.get_problem_data(solver=cp.SCS)
        param_cone_prog = data[cp.settings.PARAM_PROB]
        solver = SCS()
        raw_solution = solver.solve_via_data(
            data, warm_start=False, verbose=False, solver_opts={})['x']
        sltn_dict = param_cone_prog.split_solution(
            raw_solution, active_vars=var_dict)
        adjoint = param_cone_prog.split_adjoint(sltn_dict)
        self.assertEqual(adjoint.shape, raw_solution.shape)
        for value in sltn_dict[x.id]:
            self.assertTrue(any(value == adjoint))

        problem.solve(cp.SCS)
        self.assertItemsAlmostEqual(x.value, sltn_dict[x.id])

        # Log in constraint.
        obj = cp.Minimize(sum(x))
        constr = [cp.log(x) >= 0, x <= [1, 1]]
        problem = cp.Problem(obj, constr)
        data, _, _ = problem.get_problem_data(solver=cp.SCS)
        param_cone_prog = data[cp.settings.PARAM_PROB]
        solver = SCS()
        raw_solution = solver.solve_via_data(
            data, warm_start=False, verbose=False, solver_opts={})['x']
        sltn_dict = param_cone_prog.split_solution(
            raw_solution, active_vars=var_dict)
        adjoint = param_cone_prog.split_adjoint(sltn_dict)
        self.assertEqual(adjoint.shape, raw_solution.shape)
        for value in sltn_dict[x.id]:
            self.assertTrue(any(value == adjoint))
        self.assertItemsAlmostEqual(
            param_cone_prog.split_adjoint(sltn_dict), raw_solution)

        problem.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(x.value, sltn_dict[x.id])

    def test_psd_var(self):
        """Test PSD variable.
        """
        s = cp.Variable((2, 2), PSD=True)
        var_dict = {s.id: s}
        obj = cp.Maximize(cp.minimum(s[0, 1], 10))
        const = [cp.diag(s) == np.ones(2)]
        problem = cp.Problem(obj, const)
        data, _, _ = problem.get_problem_data(solver=cp.SCS)
        param_cone_prog = data[cp.settings.PARAM_PROB]
        solver = SCS()
        raw_solution = solver.solve_via_data(
            data, warm_start=False, verbose=False, solver_opts={})['x']
        sltn_dict = param_cone_prog.split_solution(
            raw_solution, active_vars=var_dict)
        self.assertEqual(sltn_dict[s.id].shape, s.shape)
        sltn_value = sltn_dict[s.id]
        adjoint = param_cone_prog.split_adjoint(sltn_dict)
        self.assertEqual(adjoint.shape, raw_solution.shape)
        self.assertTrue(any(sltn_value[0, 0] == adjoint))
        self.assertTrue(any(sltn_value[1, 1] == adjoint))
        # off-diagonals of adjoint will be scaled by two
        self.assertTrue(any(2 * np.isclose(sltn_value[0, 1], adjoint)))
        self.assertTrue(any(2 * np.isclose(sltn_value[1, 0], adjoint)))

        problem.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(s.value, sltn_value)
