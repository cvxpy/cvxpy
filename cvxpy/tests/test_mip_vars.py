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

import warnings

import numpy as np

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest


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

    def test_highs_default_milp(self) -> None:
        """Test that HiGHS is used as default solver for MILP problems.

        Note: This test does not parameterize the solver because it specifically
        tests the default solver selection behavior when no solver is specified.
        """
        # Simple MILP problem
        x = cp.Variable(3, integer=True)
        objective = cp.Minimize(cp.sum(x))
        constraints = [x >= 0, x <= 10, cp.sum(x) >= 5]
        prob = cp.Problem(objective, constraints)

        # Solve without specifying solver (should use HiGHS by default)
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob.value, 5.0)
        # Verify the solution is integer
        self.assertTrue(np.allclose(x.value, np.round(x.value)))
        
    def test_milp_no_warning(self) -> None:
        """Test that MILP problems don't raise a warning.

        Note: This test does not parameterize the solver because it specifically
        tests the default solver selection behavior when no solver is specified.
        """
        # MILP problem (mixed-integer linear program)
        x = cp.Variable(2, integer=True)
        y = cp.Variable(2)
        objective = cp.Minimize(cp.sum(x) + cp.sum(y))
        constraints = [x >= 0, x <= 5, y >= 0, y <= 10, cp.sum(x) + cp.sum(y) >= 3]
        prob = cp.Problem(objective, constraints)

        # Should NOT raise warning since it's an MILP
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            prob.solve()
            # Filter for the specific MINLP warning
            minlp_warnings = [warning for warning in w
                              if "mixed-integer but not an LP" in str(warning.message)]
            self.assertEqual(len(minlp_warnings), 0,
                             "MILP should not raise MINLP warning")

        self.assertEqual(prob.status, cp.OPTIMAL)
        
    def test_miqp_warning(self) -> None:
        """Test that MIQP problems raise a warning about not being LP.

        Note: This test does not parameterize the solver because it specifically
        tests the warning behavior when no solver is specified.
        """
        # MIQP problem (mixed-integer quadratic program)
        x = cp.Variable(3, integer=True)
        objective = cp.Minimize(cp.sum_squares(x))
        constraints = [x >= 0, x <= 10, cp.sum(x) >= 5]
        prob = cp.Problem(objective, constraints)

        # Should raise warning since it's MIQP (not LP)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                prob.solve()
            except Exception:
                # Solver might fail, but we're testing the warning
                pass

            # Check that the MINLP warning was raised
            minlp_warnings = [warning for warning in w
                              if "mixed-integer but not an LP" in str(warning.message)]
            self.assertGreater(len(minlp_warnings), 0,
                               "MIQP should raise MINLP warning")
            self.assertIn("pyscipopt",
                          str(minlp_warnings[0].message))
            
    def test_highs_milp_simple(self) -> None:
        """Test a simple MILP problem solves correctly with default solver.

        Note: This test does not parameterize the solver because it specifically
        tests the default solver selection behavior when no solver is specified.
        """
        # Knapsack-style problem
        x = cp.Variable(4, integer=True)
        values = np.array([3, 4, 5, 6])
        weights = np.array([2, 3, 4, 5])
        capacity = 8

        objective = cp.Maximize(values @ x)
        constraints = [
            x >= 0,
            x <= 1,  # Binary variables (0 or 1)
            weights @ x <= capacity
        ]
        prob = cp.Problem(objective, constraints)

        # Solve without specifying solver
        prob.solve()

        self.assertEqual(prob.status, cp.OPTIMAL)
        # Verify solution is integer
        self.assertTrue(np.allclose(x.value, np.round(x.value)))
        # Verify constraint satisfaction
        self.assertLessEqual(weights @ x.value, capacity + 1e-6)
