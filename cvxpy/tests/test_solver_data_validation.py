
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
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestSolverDataValidation(BaseTest):
    """Test that NaN/Inf values in problem data are caught early.

    These tests verify that NaN/Inf values in constants are detected
    during apply_parameters() before being sent to the solver.
    Note: Parameters are already validated at assignment time, so
    NaN/Inf in parameters are caught earlier.
    """

    def test_inf_constant_in_constraint(self):
        """Inf in a constraint constant should raise ValueError during solve."""
        x = cp.Variable()
        c = cp.Constant(np.inf)
        prob = cp.Problem(cp.Minimize(x), [x >= c])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_nan_constant_in_objective(self):
        """NaN in objective coefficients should raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x * cp.Constant(np.nan)), [x >= 0])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_inf_constant_in_objective(self):
        """Inf in objective coefficients should raise ValueError."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x + cp.Constant(np.inf)), [x >= 0])

        if cp.SCS in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.SCS)

    def test_valid_problem_solves(self):
        """A valid problem should still solve correctly."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 1])

        if cp.SCS in cp.installed_solvers():
            prob.solve(solver=cp.SCS)
            self.assertAlmostEqual(x.value, 1.0, places=3)

    def test_qp_solver_with_inf(self):
        """Test that QP solvers also catch Inf in problem data."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x**2), [x >= cp.Constant(np.inf)])

        if cp.OSQP in cp.installed_solvers():
            with pytest.raises(ValueError, match="contains NaN or Inf"):
                prob.solve(solver=cp.OSQP)
