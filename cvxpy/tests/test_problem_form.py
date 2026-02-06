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

import cvxpy as cp
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.problems.problem_form import ProblemForm
from cvxpy.tests.base_test import BaseTest


class TestProblemForm(BaseTest):

    def test_lp(self) -> None:
        """LP: no quadratic objective, NonNeg and Zero cones."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1, x[0] == 2])
        form = ProblemForm(prob)

        self.assertFalse(form.has_quadratic_objective())
        self.assertIn(NonNeg, form.cones())
        self.assertIn(Zero, form.cones())
        self.assertFalse(form.is_mixed_integer())
        self.assertTrue(form.has_constraints())

    def test_qp(self) -> None:
        """QP: quadratic objective."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertTrue(form.has_constraints())

    def test_qp_objective_no_soc(self) -> None:
        """sum_squares in objective is handled by QP path, no SOC needed."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        # quad_over_lin is in SOC_ATOMS but the QP path handles it,
        # so cones() should NOT include SOC.
        self.assertNotIn(SOC, form.cones())

    def test_quadratic_of_log_needs_exp_cone(self) -> None:
        """sum_squares(log(x)) has quadratic term but needs ExpCone for log."""
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.log(x))))
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(ExpCone, form.cones())
        # The outer sum_squares (quad_over_lin) is handled by QP path,
        # so SOC is NOT needed — only ExpCone for the inner log.
        self.assertNotIn(SOC, form.cones())

    def test_quadratic_of_quadratic_needs_soc(self) -> None:
        """sum_squares(sum_squares(x)) has quadratic term but needs SOC."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.sum_squares(x))))
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(SOC, form.cones())

    def test_socp(self) -> None:
        """SOCP: SOC in cones."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        form = ProblemForm(prob)

        self.assertIn(SOC, form.cones())
        self.assertTrue(form.has_constraints())

    def test_socp_constraint(self) -> None:
        """SOCP via explicit SOC constraint."""
        x = cp.Variable(2)
        t = cp.Variable()
        prob = cp.Problem(cp.Minimize(t), [SOC(t, x)])
        form = ProblemForm(prob)

        self.assertIn(SOC, form.cones())

    def test_sdp(self) -> None:
        """SDP: PSD in cones."""
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), [X >> 0])
        form = ProblemForm(prob)

        self.assertIn(PSD, form.cones())
        self.assertTrue(form.has_constraints())

    def test_psd_variable(self) -> None:
        """PSD variable without explicit PSD constraint."""
        X = cp.Variable((2, 2), PSD=True)
        prob = cp.Problem(cp.Minimize(cp.trace(X)))
        form = ProblemForm(prob)

        self.assertIn(PSD, form.cones())
        self.assertTrue(form.has_constraints())

    def test_exp_cone(self) -> None:
        """Exponential cone atoms."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= 2])
        form = ProblemForm(prob)

        self.assertIn(ExpCone, form.cones())

    def test_pow_cone(self) -> None:
        """Power cone via explicit PowCone3D constraint."""
        x = cp.Variable()
        y = cp.Variable()
        z = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [PowCone3D(x, y, z, [0.5])])
        form = ProblemForm(prob)

        self.assertIn(PowCone3D, form.cones())

    def test_mixed_integer(self) -> None:
        """Mixed-integer problem."""
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 5])
        form = ProblemForm(prob)

        self.assertTrue(form.is_mixed_integer())
        self.assertTrue(form.has_constraints())

    def test_boolean(self) -> None:
        """Boolean variable is also mixed-integer."""
        x = cp.Variable(2, boolean=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])
        form = ProblemForm(prob)

        self.assertTrue(form.is_mixed_integer())

    def test_unconstrained_linear(self) -> None:
        """Unconstrained linear objective: no cones, no constraints."""
        x = cp.Variable()
        # This is unbounded, but ProblemForm only analyzes structure.
        prob = cp.Problem(cp.Minimize(x))
        form = ProblemForm(prob)

        self.assertFalse(form.has_quadratic_objective())
        self.assertEqual(form.cones(), set())
        self.assertFalse(form.is_mixed_integer())
        self.assertFalse(form.has_constraints())

    def test_variable_domain_implies_constraints(self) -> None:
        """Variable with nonneg=True domain implies has_constraints."""
        x = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Minimize(x))
        form = ProblemForm(prob)

        # No explicit constraints or cones from atoms, but domain exists
        self.assertTrue(form.has_constraints())

    def test_exp_cone_from_atom_only(self) -> None:
        """ExpCone inferred from atom, no explicit exp cone constraint."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.exp(x)))
        form = ProblemForm(prob)

        self.assertIn(ExpCone, form.cones())
        self.assertTrue(form.has_constraints())

    def test_caching(self) -> None:
        """Results are cached across calls."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        form = ProblemForm(prob)

        cones1 = form.cones()
        cones2 = form.cones()
        self.assertIs(cones1, cones2)

    def test_solve_lp(self) -> None:
        """Verify ProblemForm analysis matches a solvable LP."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        form = ProblemForm(prob)

        self.assertFalse(form.has_quadratic_objective())
        self.assertIn(NonNeg, form.cones())
        self.assertFalse(form.is_mixed_integer())

        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.array([1.0, 1.0]))

    def test_sdp_nsd_variable(self) -> None:
        """NSD variable triggers PSD cone."""
        X = cp.Variable((2, 2), NSD=True)
        prob = cp.Problem(cp.Maximize(cp.trace(X)))
        form = ProblemForm(prob)

        self.assertIn(PSD, form.cones())

    def test_qp_with_conic_constraint(self) -> None:
        """QP objective + conic constraint: ExpCone from constraint, no SOC."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [cp.sum(cp.log(x)) >= 1])
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(ExpCone, form.cones())
        self.assertNotIn(SOC, form.cones())

    def test_mixed_objective(self) -> None:
        """sum_squares(x) + log(y): QP handles sum_squares, ExpCone for log."""
        x = cp.Variable()
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x) - cp.log(y)))
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(ExpCone, form.cones())
        self.assertNotIn(SOC, form.cones())

    def test_qp_with_soc_constraint(self) -> None:
        """QP objective + SOC constraint: SOC comes from constraint."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [cp.norm(x, 2) <= 1])
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(SOC, form.cones())

    def test_non_quad_objective(self) -> None:
        """Non-quadratic objective: no QP filtering, normal cone detection."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        form = ProblemForm(prob)

        self.assertFalse(form.has_quadratic_objective())
        self.assertIn(SOC, form.cones())
        self.assertIn(NonNeg, form.cones())
