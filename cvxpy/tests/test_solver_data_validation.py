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
    """Test that NaN values in problem data are caught."""

    def test_nan_in_parameter_raises(self):
        """NaN in a parameter value should raise ValueError at assignment."""
        p = cp.Parameter()
        with pytest.raises(ValueError, match="must be real"):
            p.value = np.nan

    def test_nan_constant_in_objective(self):
        """NaN in objective should raise ValueError during solve."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x * np.nan), [x >= 0])
        with pytest.raises(ValueError, match="contains NaN"):
            prob.solve(solver=cp.SCS)

    def test_nan_constant_in_constraint(self):
        """NaN in constraint should raise ValueError during solve."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= np.nan])
        with pytest.raises(ValueError, match="contains NaN"):
            prob.solve(solver=cp.SCS)

    def test_inf_in_constraint_rhs_allowed(self):
        """Inf in constraint RHS (b vector) should be allowed."""
        x = cp.Variable()
        # x <= inf is trivially satisfied, should not raise
        prob = cp.Problem(cp.Minimize(x), [x >= 1, x <= np.inf])
        prob.solve(solver=cp.OSQP)
        self.assertAlmostEqual(x.value, 1.0, places=3)

    def test_inf_constant_in_objective(self):
        """A valid problem should still solve correctly."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(np.inf * x), [x >= 1])
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            prob.solve(solver=cp.SCS)
