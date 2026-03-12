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

import cvxpy as cp
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, Zero
from cvxpy.error import SolverError
from cvxpy.problems.problem_form import ProblemForm, pick_default_solver
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL as ClarabelSolver
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS as ScsSolver
from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP as OsqpSolver
from cvxpy.reductions.solvers.solving_chain import resolve_and_build_chain
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

    def test_qp_filtering(self) -> None:
        """QP path filters SOC from quad atoms but preserves other cones."""
        # sum_squares alone: QP path handles it, no SOC needed
        x = cp.Variable()
        form = ProblemForm(cp.Problem(cp.Minimize(cp.sum_squares(x))))
        self.assertTrue(form.has_quadratic_objective())
        self.assertNotIn(SOC, form.cones(quad_obj=True))

        # sum_squares(log(x)): QP handles outer quad, ExpCone for log, no SOC
        x = cp.Variable(pos=True)
        form = ProblemForm(
            cp.Problem(cp.Minimize(cp.sum_squares(cp.log(x)))))
        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(ExpCone, form.cones(quad_obj=True))
        self.assertNotIn(SOC, form.cones(quad_obj=True))

        # sum_squares(sum_squares(x)): inner quad needs SOC even with QP path
        x = cp.Variable()
        form = ProblemForm(
            cp.Problem(cp.Minimize(cp.sum_squares(cp.sum_squares(x)))))
        self.assertTrue(form.has_quadratic_objective())
        self.assertIn(SOC, form.cones())

    def test_quad_over_lin_constant_denom_qp_path(self) -> None:
        """quad_over_lin with non-PWL numerator and constant denominator.

        Regression for dcp2cone fix (25ef23da9): the old guard in
        Dcp2Cone.canonicalize_expr used expr.is_qpwa(), which requires the
        numerator to be piecewise-linear.  For quad_over_lin(exp(x), 1),
        exp(x) is not PWL so is_qpwa() returned False, making Dcp2Cone
        fall through to cone canon (SOC) even though the denominator is
        constant and the quad canon is valid.

        The fix checks only args[1].is_constant() (the canonicalized
        denominator).  We verify that Dcp2Cone produces ExpCone (for exp)
        but no SOC (quad_over_lin handled by QP path).
        """
        from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone

        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(cp.exp(x), 1)))
        dcp2cone = Dcp2Cone(quad_obj=True)
        new_prob, _ = dcp2cone.apply(prob)
        constr_types = {type(c) for c in new_prob.constraints}
        self.assertIn(ExpCone, constr_types)
        self.assertNotIn(SOC, constr_types)

    def test_socp(self) -> None:
        """SOCP: SOC in cones."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        form = ProblemForm(prob)

        self.assertIn(SOC, form.cones())
        self.assertTrue(form.has_constraints())

    def test_sdp(self) -> None:
        """SDP: PSD in cones."""
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), [X >> 0])
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
        """Mixed-integer problem (integer and boolean)."""
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 5])
        form = ProblemForm(prob)
        self.assertTrue(form.is_mixed_integer())

        y = cp.Variable(2, boolean=True)
        form2 = ProblemForm(cp.Problem(cp.Minimize(cp.sum(y)), [y >= 0]))
        self.assertTrue(form2.is_mixed_integer())

    def test_unconstrained_linear(self) -> None:
        """Unconstrained linear objective: no cones, no constraints."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x))
        form = ProblemForm(prob)

        self.assertFalse(form.has_quadratic_objective())
        self.assertEqual(form.cones(), set())
        self.assertFalse(form.is_mixed_integer())
        self.assertFalse(form.has_constraints())

    def test_sdp_nsd_variable(self) -> None:
        """NSD variable triggers PSD cone."""
        X = cp.Variable((2, 2), NSD=True)
        prob = cp.Problem(cp.Maximize(cp.trace(X)))
        form = ProblemForm(prob)

        self.assertIn(PSD, form.cones())


class TestPickDefaultSolver(BaseTest):
    """Tests for pick_default_solver()."""

    def _is_commercial(self, solver) -> bool:
        return solver is not None and solver.name() in slv_def.COMMERCIAL_SOLVERS

    def test_lp_gets_clarabel(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        solver = pick_default_solver(ProblemForm(prob))
        self.assertTrue(
            self._is_commercial(solver) or isinstance(solver, ClarabelSolver))

    def test_qp_gets_osqp(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 1])
        solver = pick_default_solver(ProblemForm(prob))
        self.assertTrue(
            self._is_commercial(solver) or isinstance(solver, OsqpSolver))

    def test_sdp_gets_scs(self) -> None:
        X = cp.Variable((2, 2), symmetric=True)
        prob = cp.Problem(cp.Minimize(cp.lambda_max(X)), [X >> 0])
        solver = pick_default_solver(ProblemForm(prob))
        self.assertTrue(
            self._is_commercial(solver) or isinstance(solver, ScsSolver))

    def test_socp_gets_clarabel(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 2)), [x >= 1])
        solver = pick_default_solver(ProblemForm(prob))
        self.assertTrue(
            self._is_commercial(solver) or isinstance(solver, ClarabelSolver))

    def test_mi_gets_highs_or_premium(self) -> None:
        from cvxpy.reductions.solvers.conic_solvers.highs_conif import HIGHS as HighsSolver
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 5])
        solver = pick_default_solver(ProblemForm(prob))
        self.assertTrue(
            self._is_commercial(solver) or isinstance(solver, HighsSolver)
            or solver is None)


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

    def test_fallback_when_default_missing(self) -> None:
        """Mock pick_default_solver to return None, verify fallback warns."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        with patch(
            "cvxpy.reductions.solvers.solving_chain.pick_default_solver",
            return_value=None,
        ) as mock_pick:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                chain = resolve_and_build_chain(prob, solver=None)
                mock_pick.assert_called_once()
                warn_msgs = [str(wi.message) for wi in w]
                self.assertTrue(
                    any("default solvers" in m.lower() for m in warn_msgs),
                    f"Expected a warning about default solvers, got: {warn_msgs}"
                )
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0])


class TestGP(BaseTest):
    """Tests for GP-aware ProblemForm and solving chain."""

    def test_gp_cones(self) -> None:
        """GP cone detection: posynomial needs ExpCone, geo_mean does not."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        form = ProblemForm(
            cp.Problem(cp.Minimize(x + y), [x * y >= 1]), gp=True)
        self.assertIn(ExpCone, form.cones())
        # GP has no QP path: cones(quad_obj=True) == cones()
        self.assertEqual(form.cones(quad_obj=True), form.cones())

        x2 = cp.Variable(2, pos=True)
        form2 = ProblemForm(
            cp.Problem(cp.Maximize(cp.geo_mean(x2)), [x2 <= 2]), gp=True)
        self.assertNotIn(ExpCone, form2.cones())

    def test_gp_solve_default(self) -> None:
        """GP + solver=None -> solves correctly."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x + y), [x * y >= 1])
        chain = resolve_and_build_chain(prob, solver=None, gp=True)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertAlmostEqual(prob.value, 2.0, places=3)

    def test_gp_solve_named(self) -> None:
        """GP + 'SCS' -> works."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x + y), [x * y >= 1])
        chain = resolve_and_build_chain(prob, solver="SCS", gp=True)
        self.assertEqual(chain.solver.name(), cp.SCS)
        soln = chain.solve(prob, warm_start=False, verbose=False, solver_opts={})
        prob.unpack(soln)
        self.assertAlmostEqual(prob.value, 2.0, places=2)

    def test_gp_custom_solver(self) -> None:
        """GP + custom ConicSolver -> works."""
        class MyConicSolver(ScsSolver):
            def name(self) -> str:
                return "MY_GP_SOLVER"

        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x + y), [x * y >= 1])
        chain = resolve_and_build_chain(prob, solver=MyConicSolver(), gp=True)
        self.assertEqual(chain.solver.name(), "MY_GP_SOLVER")
