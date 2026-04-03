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
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif(
    'IPOPT' not in INSTALLED_SOLVERS,
    reason='IPOPT is not installed.'
)
class TestDNLPDualRecovery:
    """Test that dual variables are correctly recovered from NLP solvers.

    These tests verify that the constraint multipliers returned by the
    NLP solver are properly sliced, reshaped, and propagated back through
    the reduction chain so that ``constraint.dual_value`` is populated
    after a solve.
    """

    def test_lp_dual_values(self):
        """Verify duals on a simple LP solved via IPOPT.

        minimize   -4*x[0] - 5*x[1]
        subject to  2*x[0] +   x[1] <= 3
                      x[0] + 2*x[1] <= 3
                    x >= 0

        Optimal: x = [1, 1], obj = -9
        LP duals: lambda_1 = 1, lambda_2 = 2
        """
        x = cp.Variable(2, nonneg=True)
        c1 = 2 * x[0] + x[1] <= 3
        c2 = x[0] + 2 * x[1] <= 3
        prob = cp.Problem(cp.Minimize(-4 * x[0] - 5 * x[1]), [c1, c2])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, [1.0, 1.0], atol=1e-5)

        # Dual variables should be populated (not None).
        assert c1.dual_value is not None
        assert c2.dual_value is not None

        # LP duals for this problem are 1 and 2.
        np.testing.assert_allclose(c1.dual_value, 1.0, atol=1e-4)
        np.testing.assert_allclose(c2.dual_value, 2.0, atol=1e-4)

    def test_equality_constraint_dual(self):
        """Verify dual recovery for an equality constraint.

        minimize   (x - 3)^2
        subject to  x == 1

        Optimal: x = 1, obj = 4
        KKT stationarity: 2*(x - 3) + lambda = 0  =>  lambda = 4
        """
        x = cp.Variable()
        c_eq = x == 1
        prob = cp.Problem(cp.Minimize((x - 3) ** 2), [c_eq])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, 1.0, atol=1e-5)
        assert c_eq.dual_value is not None
        np.testing.assert_allclose(c_eq.dual_value, 4.0, atol=1e-4)

    def test_smooth_nlp_duals_nonzero(self):
        """Verify that duals are populated for a smooth NLP.

        minimize   exp(x) + exp(y)
        subject to  x + y == 1

        At optimum x = y = 0.5,  dual = -exp(0.5) ≈ -1.6487
        (stationarity: exp(x) + lambda = 0 => lambda = -exp(0.5))
        """
        x = cp.Variable()
        y = cp.Variable()
        c_eq = x + y == 1
        prob = cp.Problem(cp.Minimize(cp.exp(x) + cp.exp(y)), [c_eq])
        x.value = 0.5
        y.value = 0.5
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, 0.5, atol=1e-5)
        assert c_eq.dual_value is not None
        np.testing.assert_allclose(
            c_eq.dual_value, -np.exp(0.5), atol=1e-4
        )

    def test_multiple_constraint_shapes(self):
        """Duals are correctly reshaped for vector constraints."""
        x = cp.Variable(3)
        c_ineq = x <= 2
        c_eq = cp.sum(x) == 3
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [c_ineq, c_eq])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        # x should be [1, 1, 1]
        np.testing.assert_allclose(x.value, [1.0, 1.0, 1.0], atol=1e-5)

        # Inequality not active => dual should be ~0
        assert c_ineq.dual_value is not None
        assert c_ineq.dual_value.shape == (3,)
        np.testing.assert_allclose(c_ineq.dual_value, 0.0, atol=1e-4)

        # Equality dual should be non-None
        assert c_eq.dual_value is not None

    def test_no_constraints_no_crash(self):
        """Unconstrained problem: no dual variables to recover."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize((x - 5) ** 2))
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, 5.0, atol=1e-5)

    def test_esr_atom_user_duals_propagated(self):
        """Duals propagate through ESR (abs) canonicalization.

        The abs atom introduces auxiliary constraints internally.
        Only user-visible constraint duals should be populated.
        """
        x = cp.Variable()
        c_eq = x == -2
        prob = cp.Problem(cp.Minimize(cp.abs(x)), [c_eq])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(x.value, -2.0, atol=1e-5)
        # The user constraint should have a dual value.
        assert c_eq.dual_value is not None
