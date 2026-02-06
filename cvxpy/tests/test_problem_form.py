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
from unittest.mock import patch

import numpy as np

import cvxpy as cp
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, PowConeND, Zero
from cvxpy.error import SolverError
from cvxpy.problems.problem_form import ProblemForm, pick_default_solver
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as ClarabelSolver
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS as EcosSolver
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as ScsSolver
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OsqpSolver
from cvxpy.reductions.solvers.solving_chain import build_solving_chain, resolve_and_build_chain
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


class TestPickDefaultSolver(BaseTest):
    """Tests for pick_default_solver().

    Premium solvers (MOSEK, MOREAU, GUROBI) are checked first by
    pick_default_solver. When any of them is installed, it wins for all
    problems it can handle. The tests below account for this by accepting
    premium solvers as valid results alongside the expected open-source
    default.
    """

    # Premium solver classes that may be installed and take priority.
    PREMIUM_SOLVER_CLASSES: tuple[type, ...] = ()

    @classmethod
    def setUpClass(cls) -> None:
        from cvxpy.reductions.solvers.conic_solvers.gurobi_conif import GUROBI as GurobiSolver
        from cvxpy.reductions.solvers.conic_solvers.moreau_conif import MOREAU as MoreauSolver
        from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK as MosekSolver
        cls.PREMIUM_SOLVER_CLASSES = (MosekSolver, MoreauSolver, GurobiSolver)

    def _is_premium(self, solver) -> bool:
        return isinstance(solver, self.PREMIUM_SOLVER_CLASSES)

    def test_lp_gets_clarabel(self) -> None:
        """LP goes to Clarabel (or a premium solver)."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, ClarabelSolver))

    def test_qp_gets_osqp(self) -> None:
        """QP goes to OSQP (or a premium solver)."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, OsqpSolver))

    def test_sdp_gets_scs(self) -> None:
        """SDP goes to SCS (or a premium solver)."""
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), [X >> 0])
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, ScsSolver))

    def test_socp_gets_clarabel(self) -> None:
        """SOCP goes to Clarabel (or a premium solver)."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, ClarabelSolver))

    def test_exp_cone_gets_clarabel(self) -> None:
        """ExpCone problem goes to Clarabel (or a premium solver)."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= 2])
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, ClarabelSolver))

    def test_unconstrained_lp(self) -> None:
        """Unconstrained linear objective is an LP (or premium solver)."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x))
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, ClarabelSolver))

    def test_unconstrained_qp(self) -> None:
        """Unconstrained QP → OSQP (or a premium solver)."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, OsqpSolver))

    def test_mi_gets_highs_or_premium(self) -> None:
        """MI problem goes to HIGHS (or a premium solver)."""
        from cvxpy.reductions.solvers.conic_solvers.highs_conif import HIGHS as HighsSolver
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 5])
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        # Premium MI-capable solvers (MOSEK, GUROBI) can also handle this.
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, HighsSolver)
            or solver is None)

    def test_psd_variable_gets_scs(self) -> None:
        """PSD variable (implies PSD cone) → SCS (or premium solver)."""
        X = cp.Variable((2, 2), PSD=True)
        prob = cp.Problem(cp.Minimize(cp.trace(X)))
        form = ProblemForm(prob)
        solver = pick_default_solver(form)
        self.assertTrue(
            self._is_premium(solver) or isinstance(solver, ScsSolver))

    def test_always_returns_solver(self) -> None:
        """pick_default_solver always returns a solver for basic problems."""
        problems = [
            # LP
            cp.Problem(cp.Minimize(cp.sum(cp.Variable(2))),
                       [cp.Variable(2) >= 0]),
            # QP
            cp.Problem(cp.Minimize(cp.sum_squares(cp.Variable(2))),
                       [cp.Variable(2) >= 0]),
            # SOCP
            cp.Problem(cp.Minimize(cp.norm(cp.Variable(2), 2)),
                       [cp.Variable(2) >= 0]),
        ]
        for prob in problems:
            form = ProblemForm(prob)
            solver = pick_default_solver(form)
            self.assertIsNotNone(solver)


class TestBuildSolvingChain(BaseTest):
    """Tests for build_solving_chain()."""

    def test_lp_with_clarabel(self) -> None:
        """Build and solve an LP with Clarabel."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        chain = build_solving_chain(prob, ClarabelSolver())
        self.assertEqual(chain.solver.name(), cp.CLARABEL)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])

    def test_socp_with_clarabel(self) -> None:
        """Build and solve a SOCP with Clarabel."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        chain = build_solving_chain(prob, ClarabelSolver())
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])

    def test_qp_with_osqp(self) -> None:
        """Build and solve a QP with OSQP."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        chain = build_solving_chain(prob, OsqpSolver())
        self.assertEqual(chain.solver.name(), cp.OSQP)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])

    def test_sdp_with_scs(self) -> None:
        """Build and solve an SDP with SCS."""
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.trace(X)), [X >> np.eye(2)])
        chain = build_solving_chain(prob, ScsSolver())
        self.assertEqual(chain.solver.name(), cp.SCS)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertAlmostEqual(prob.value, 2.0, places=3)

    def test_exp_cone_with_clarabel(self) -> None:
        """Build and solve an exponential cone problem."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.log(x))), [x <= 2])
        chain = build_solving_chain(prob, ClarabelSolver())
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [2.0, 2.0], places=3)

    def test_pow_cone_nd_with_scs(self) -> None:
        """PowConeND gets exact conversion to PowCone3D for SCS."""
        x = cp.Variable(3, nonneg=True)
        z = cp.Variable()
        prob = cp.Problem(cp.Maximize(z),
                          [PowConeND(x, z, np.array([0.3, 0.3, 0.4])),
                           x <= 1])
        chain = build_solving_chain(prob, ScsSolver())
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertAlmostEqual(z.value, 1.0, places=3)

    def test_constant_problem(self) -> None:
        """Problem with 0 variables uses ConstantSolver."""
        from cvxpy.reductions.solvers.constant_solver import ConstantSolver
        prob = cp.Problem(cp.Minimize(cp.Constant(42)))
        chain = build_solving_chain(prob, ClarabelSolver())
        self.assertIsInstance(chain.solver, ConstantSolver)

    def test_maximize_flipped(self) -> None:
        """Maximize objectives are flipped correctly."""
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(x), [x <= 5])
        chain = build_solving_chain(prob, ClarabelSolver())
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertAlmostEqual(x.value, 5.0, places=3)

    def test_with_problem_form(self) -> None:
        """Passing a pre-computed ProblemForm works."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        form = ProblemForm(prob)
        chain = build_solving_chain(prob, ClarabelSolver(),
                                    problem_form=form)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])


class TestResolveAndBuildChain(BaseTest):
    """Tests for resolve_and_build_chain()."""

    def test_solver_none_picks_default(self) -> None:
        """solver=None, LP -> solves correctly."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        chain = resolve_and_build_chain(prob, solver=None)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])

    def test_solver_string_clarabel(self) -> None:
        """solver='CLARABEL', SOCP -> solves correctly."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        chain = resolve_and_build_chain(prob, solver="CLARABEL")
        self.assertEqual(chain.solver.name(), cp.CLARABEL)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])

    def test_solver_string_osqp(self) -> None:
        """solver='OSQP', QP -> solves correctly."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        chain = resolve_and_build_chain(prob, solver="OSQP")
        self.assertEqual(chain.solver.name(), cp.OSQP)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])

    def test_solver_string_case_insensitive(self) -> None:
        """solver='clarabel' (lowercase) works."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        chain = resolve_and_build_chain(prob, solver="clarabel")
        self.assertEqual(chain.solver.name(), cp.CLARABEL)

    def test_solver_not_installed(self) -> None:
        """solver='NONEXISTENT' -> SolverError."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        with self.assertRaises(SolverError):
            resolve_and_build_chain(prob, solver="NONEXISTENT")

    def test_solver_cannot_handle(self) -> None:
        """solver='OSQP', SOCP -> SolverError."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        with self.assertRaises(SolverError):
            resolve_and_build_chain(prob, solver="OSQP")

    def test_custom_solver_instance(self) -> None:
        """Solver instance passed directly -> works."""
        class MyConicSolver(ScsSolver):
            def name(self) -> str:
                return "MY_CUSTOM_SOLVER"

        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        chain = resolve_and_build_chain(prob, solver=MyConicSolver())
        self.assertEqual(chain.solver.name(), "MY_CUSTOM_SOLVER")

    def test_solver_string_dual_map_prefers_qp(self) -> None:
        """solver='HIGHS', QP problem -> picks QP instance."""
        from cvxpy.reductions.solvers.qp_solvers.highs_qpif import HIGHS as HighsQpSolver
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        chain = resolve_and_build_chain(prob, solver="HIGHS")
        # Should pick the QP instance for a quadratic problem.
        self.assertIsInstance(chain.solver, HighsQpSolver)

    def test_fallback_when_default_missing(self) -> None:
        """Mock pick_default_solver to return None, verify fallback warns."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        with patch(
            "cvxpy.problems.problem_form.pick_default_solver",
            return_value=None,
        ) as mock_pick:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                chain = resolve_and_build_chain(prob, solver=None)
                mock_pick.assert_called_once()
                # Should have issued a warning about default solvers.
                warn_msgs = [str(wi.message) for wi in w]
                self.assertTrue(
                    any("default solvers" in m.lower() for m in warn_msgs),
                    f"Expected a warning about default solvers, got: {warn_msgs}"
                )
        # Chain should still be functional.
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])
