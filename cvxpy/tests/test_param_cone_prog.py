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

import numpy as np

import cvxpy as cp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.tests.base_test import BaseTest


class TestParamConeProg(BaseTest):
    # Overridden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places: int = 2) -> None:
        super(TestParamConeProg, self).assertItemsAlmostEqual(a, b, places=places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places: int = 2) -> None:
        super(TestParamConeProg, self).assertAlmostEqual(a, b, places=places)

    def test_log_problem(self) -> None:
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

    def test_psd_var(self) -> None:
        """Test PSD variable.

        PSD implies symmetric, so CvxAttr2Constr creates a reduced variable
        (upper triangle) with its own ID.  Use the solving chain to invert
        the raw solver output back to the original variable.
        """
        s = cp.Variable((2, 2), PSD=True)
        obj = cp.Maximize(cp.minimum(s[0, 1], 10))
        const = [cp.diag(s) == np.ones(2)]
        problem = cp.Problem(obj, const)
        data, chain, inverse_data = problem.get_problem_data(solver=cp.SCS)
        solver = SCS()
        raw_result = solver.solve_via_data(
            data, warm_start=False, verbose=False, solver_opts={})

        # Invert through the full reduction chain to recover original vars.
        solution = chain.invert(raw_result, inverse_data)
        self.assertEqual(solution.primal_vars[s.id].shape, s.shape)
        sltn_value = solution.primal_vars[s.id]

        # Verify full solve matches the chain-inverted solution.
        problem.solve(solver=cp.SCS, eps=1e-5)
        self.assertItemsAlmostEqual(s.value, sltn_value)

    def test_var_bounds(self) -> None:
        """Test that lower and upper bounds on variables are propagated."""
        # Create a solver instance where bounded variables are disabled.
        solver_instance = SCS()
        solver_instance.name = lambda: "Custom SCS, no bounded variables"
        solver_instance.BOUNDED_VARIABLES = False

        lower_bounds = -10
        upper_bounds = np.arange(6).reshape((3, 2))
        x = cp.Variable((3, 2), bounds=[lower_bounds, upper_bounds])
        problem = cp.Problem(cp.Minimize(cp.sum(x)))
        data, _, _ = problem.get_problem_data(solver=solver_instance)
        param_cone_prog = data[cp.settings.PARAM_PROB]

        assert param_cone_prog.lower_bounds is None
        assert param_cone_prog.upper_bounds is None

        # Create a solver instance where bounded variables are enabled.
        solver_instance = SCS()
        solver_instance.name = lambda: "Custom SCS, bounded variables"
        solver_instance.BOUNDED_VARIABLES = True

        lower_bounds = -10
        upper_bounds = np.arange(6).reshape((3, 2))
        x = cp.Variable((3, 2), bounds=[lower_bounds, upper_bounds])
        problem = cp.Problem(cp.Minimize(cp.sum(x)))
        data, _, _ = problem.get_problem_data(solver=solver_instance)
        param_cone_prog = data[cp.settings.PARAM_PROB]

        assert np.all(param_cone_prog.lower_bounds == lower_bounds)
        param_upper_bound = np.reshape(param_cone_prog.upper_bounds, (3, 2), order="F")
        assert np.all(param_upper_bound == upper_bounds)

    def test_highs_var_bounds(self) -> None:
        """Testing variable bounds problem with HiGHS."""
        x1 = cp.Variable(bounds=[-1, 1])
        x2 = cp.Variable(bounds=[-0.5, 1])
        x3 = cp.Variable()
        objective = x1 + x2 + x3
        constraints = [-3 <= x1 + x2, x1 + x2 <= 3,
                        -4 <= x1 - x2, x1 - x2 <= 4, x3 >= -2]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        data, _, _ = prob.get_problem_data(solver=cp.HIGHS)
        param_cone_prog = data[cp.settings.PARAM_PROB]

        assert np.all(param_cone_prog.lower_bounds == np.array([-1, -0.5, -np.inf]))
        assert np.all(param_cone_prog.upper_bounds == np.array([1, 1, np.inf]))

        prob.solve(solver=cp.HIGHS)
        assert np.isclose(x1.value, -1)
        assert np.isclose(x2.value, -0.5)
        assert np.isclose(x3.value, -2)
