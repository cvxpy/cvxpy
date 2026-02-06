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
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, PowConeND, Zero
from cvxpy.problems.problem_form import ProblemForm
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as ClarabelSolver
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS as EcosSolver
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as ScsSolver
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OsqpSolver
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
        # Default cones() returns the conservative full set (includes SOC).
        self.assertIn(SOC, form.cones())
        # quad_over_lin is in SOC_ATOMS but the QP path handles it,
        # so cones(quad_obj=True) should NOT include SOC.
        self.assertNotIn(SOC, form.cones(quad_obj=True))

    def test_quadratic_of_log_needs_exp_cone(self) -> None:
        """sum_squares(log(x)) has quadratic term but needs ExpCone for log."""
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(cp.log(x))))
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(ExpCone, form.cones())
        self.assertIn(ExpCone, form.cones(quad_obj=True))
        # The outer sum_squares (quad_over_lin) is handled by QP path,
        # so cones(quad_obj=True) should NOT include SOC.
        self.assertNotIn(SOC, form.cones(quad_obj=True))

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

        cones_q1 = form.cones(quad_obj=True)
        cones_q2 = form.cones(quad_obj=True)
        self.assertIs(cones_q1, cones_q2)

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
        self.assertIn(ExpCone, form.cones(quad_obj=True))
        self.assertNotIn(SOC, form.cones(quad_obj=True))

    def test_mixed_objective(self) -> None:
        """sum_squares(x) + log(y): QP handles sum_squares, ExpCone for log."""
        x = cp.Variable()
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x) - cp.log(y)))
        form = ProblemForm(prob)

        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(ExpCone, form.cones())
        self.assertIn(ExpCone, form.cones(quad_obj=True))
        self.assertNotIn(SOC, form.cones(quad_obj=True))

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


class TestCanSolve(BaseTest):
    """Tests for Solver.can_solve(problem_form)."""

    def test_lp_clarabel(self) -> None:
        """LP is solvable by Clarabel."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        form = ProblemForm(prob)
        self.assertTrue(ClarabelSolver().can_solve(form))

    def test_socp_clarabel(self) -> None:
        """SOCP is solvable by Clarabel."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        form = ProblemForm(prob)
        self.assertTrue(ClarabelSolver().can_solve(form))

    def test_socp_not_qp_solver(self) -> None:
        """SOCP is not solvable by a QP solver (OSQP)."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        form = ProblemForm(prob)
        self.assertFalse(OsqpSolver().can_solve(form))

    def test_sdp_not_ecos(self) -> None:
        """SDP is not solvable by ECOS (no PSD support)."""
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), [X >> 0])
        form = ProblemForm(prob)
        self.assertFalse(EcosSolver().can_solve(form))

    def test_sdp_clarabel(self) -> None:
        """SDP is solvable by Clarabel."""
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), [X >> 0])
        form = ProblemForm(prob)
        self.assertTrue(ClarabelSolver().can_solve(form))

    def test_mip_rejected_by_non_mip_solver(self) -> None:
        """MIP rejected by non-MIP solver (Clarabel)."""
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 5])
        form = ProblemForm(prob)
        self.assertFalse(ClarabelSolver().can_solve(form))

    def test_requires_constr_respected(self) -> None:
        """SCS needs constraints (REQUIRES_CONSTR=True)."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x))
        form = ProblemForm(prob)
        # SCS requires constraints and this problem has none.
        self.assertFalse(ScsSolver().can_solve(form))

    def test_requires_constr_with_constraints(self) -> None:
        """SCS can solve when constraints are present."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        form = ProblemForm(prob)
        self.assertTrue(ScsSolver().can_solve(form))

    def test_qp_solver_accepts_qp(self) -> None:
        """QP solver accepts simple QP."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        form = ProblemForm(prob)
        self.assertTrue(OsqpSolver().can_solve(form))

    def test_qp_solver_rejects_socp(self) -> None:
        """QP solver rejects problem needing SOC from constraint."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)),
                          [cp.norm(x, 2) <= 1])
        form = ProblemForm(prob)
        self.assertFalse(OsqpSolver().can_solve(form))

    def test_pow_cone_nd_via_exact_conversion(self) -> None:
        """PowConeND problem solvable by Clarabel via exact conversion."""
        x = cp.Variable(3, nonneg=True)
        z = cp.Variable()
        prob = cp.Problem(cp.Maximize(z),
                          [PowConeND(x, z, np.array([0.3, 0.3, 0.4])),
                           x <= 1])
        form = ProblemForm(prob)
        # Clarabel supports PowConeND directly.
        self.assertTrue(ClarabelSolver().can_solve(form))

    def test_pow_cone_nd_exact_expansion(self) -> None:
        """PowConeND expands to PowCone3D for solvers without PowConeND."""
        x = cp.Variable(3, nonneg=True)
        z = cp.Variable()
        prob = cp.Problem(cp.Maximize(z),
                          [PowConeND(x, z, np.array([0.3, 0.3, 0.4])),
                           x <= 1])
        form = ProblemForm(prob)
        # SCS supports PowCone3D but not PowConeND; exact conversion applies.
        self.assertTrue(ScsSolver().can_solve(form))

    def test_exp_cone_not_qp_solver(self) -> None:
        """ExpCone problem not solvable by QP solver."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= 2])
        form = ProblemForm(prob)
        self.assertFalse(OsqpSolver().can_solve(form))

    def test_exp_cone_ecos(self) -> None:
        """ExpCone problem solvable by ECOS."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= 2])
        form = ProblemForm(prob)
        self.assertTrue(EcosSolver().can_solve(form))
