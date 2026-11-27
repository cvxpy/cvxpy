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


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestIPOPTDuals:

    def test_div_inequality_duals(self):
        """Inequality duals pass through the auxiliary variable div_canon adds.

        At the optimum of min x/y with x >= 1 and y <= 2, both constraints
        are active at x = 1, y = 2. The duals equal the objective's
        sensitivities: d(x/y)/dx = 1/y = 0.5 and -d(x/y)/dy = x/y**2 = 0.25.
        """
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x / y), [x >= 1, y <= 2])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(prob.constraints[0].dual_value, 0.5, rtol=1e-5)
        np.testing.assert_allclose(prob.constraints[1].dual_value, 0.25, rtol=1e-5)

    def test_equality_dual_sign(self):
        """An equality dual follows CVXPY's sign convention.

        The problem is min x/y with x == 1 and y <= 2. The Lagrangian is
        f + lambda * (x - 1), so stationarity in x gives
        lambda = -1/y = -0.5. Conic solvers report the same sign.
        """
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x / y), [x == 1, y <= 2])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(prob.constraints[0].dual_value, -0.5, rtol=1e-5)
        np.testing.assert_allclose(prob.constraints[1].dual_value, 0.25, rtol=1e-5)

    def test_log_dual_matches_conic_solver(self):
        """The dual of a single active bound matches a conic solve.

        The problem is min -log(x) with x <= 2. Its dual is 1/x = 0.5, the
        value Clarabel returns for the same convex problem.
        """
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(-cp.log(x)), [x <= 2])
        prob.solve(solver=cp.IPOPT, nlp=True)
        ipopt_dual = prob.constraints[0].dual_value
        prob.solve(solver=cp.CLARABEL)

        np.testing.assert_allclose(ipopt_dual, 0.5, rtol=1e-5)
        np.testing.assert_allclose(ipopt_dual, prob.constraints[0].dual_value, rtol=1e-4)

    def test_vector_duals_for_maximize(self):
        """Vector constraint duals land in the right entries for a maximization.

        The problem is max sum(log(x)) with x <= b, for b = [1, 2, 4]. Every
        bound is active, and entry i of the dual is 1/b_i. Distinct values
        per entry would expose a slicing or ordering error, and positive
        values check the sign after FlipObjective.
        """
        b = np.array([1.0, 2.0, 4.0])
        x = cp.Variable(3, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= b])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        dual = prob.constraints[0].dual_value
        assert dual.shape == (3,)
        np.testing.assert_allclose(dual, 1 / b, rtol=1e-5)

    def test_matrix_constraint_dual_shape(self):
        """A matrix constraint's dual comes back in the constraint's shape.

        The problem is min sum(-log(X)) with X <= B, for a 2x2 B with distinct
        entries. Every entry is active, so the dual is 1/B elementwise. This
        checks the column-major reshape.
        """
        B = np.array([[1.0, 2.0], [4.0, 8.0]])
        X = cp.Variable((2, 2), pos=True)
        prob = cp.Problem(cp.Minimize(cp.sum(-cp.log(X))), [X <= B])
        prob.solve(solver=cp.IPOPT, nlp=True)

        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(prob.constraints[0].dual_value, 1 / B, rtol=1e-5)
