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
import cvxpy.settings as s
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


def _has_diffengine(chain):
    """True iff a DIFFENGINE-backend ConeMatrixStuffing is in the chain."""
    return any(isinstance(r, ConeMatrixStuffing)
               and r.canon_backend == s.DIFFENGINE_CANON_BACKEND
               for r in chain.reductions)


def _has_eval_params(chain):
    return any(isinstance(r, EvalParams) for r in chain.reductions)


class TestIgnoreDppCorrectness(BaseTest):
    def test_two_solves_with_different_params_match_baseline(self) -> None:
        """ignore_dpp=True must produce the same answers as the DPP baseline
        across consecutive solves with different parameter values."""
        rng = np.random.default_rng(0)
        n = 5
        A_val = rng.standard_normal((n, n))
        b_val = rng.standard_normal(n)

        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        x = cp.Variable(n)
        prob_de = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))
        prob_baseline = cp.Problem(
            cp.Minimize(cp.sum_squares(A @ x - b)))

        for trial in range(2):
            A.value = A_val + 0.01 * rng.standard_normal((n, n))
            b.value = b_val + 0.01 * rng.standard_normal(n)

            prob_de.solve(solver=SOLVER, ignore_dpp=True)
            x_de = x.value.copy()

            prob_baseline.solve(solver=SOLVER)
            x_base = x.value.copy()

            self.assertItemsAlmostEqual(x_de, x_base, places=4)

    def test_maximize_obj_sign(self) -> None:
        """FlipObjective runs before stuffing; invert must flip opt_val back."""
        x = cp.Variable()
        a = cp.Parameter()
        a.value = 1.0
        prob = cp.Problem(cp.Maximize(a * x), [x <= 2.0, x >= 0])
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertAlmostEqual(prob.value, 2.0, places=4)

    def test_no_parameter_problem(self) -> None:
        """A problem with no parameters must still solve under ignore_dpp=True."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=4)


class TestIgnoreDppCaching(BaseTest):
    def test_dcp2cone_runs_once_across_two_solves(self) -> None:
        """When the diffengine path is selected, Dcp2Cone.apply should run once."""
        n = 3
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        A.value = np.eye(n)
        b.value = np.ones(n)
        x = cp.Variable(n)
        # Affine objective so the diffengine path accepts and we hit the cache.
        prob = cp.Problem(cp.Minimize(cp.sum(A @ x - b)), [x >= 0, x <= 1])

        orig_apply = Dcp2Cone.apply
        calls = []

        def counting_apply(self, problem):
            calls.append(1)
            return orig_apply(self, problem)

        Dcp2Cone.apply = counting_apply
        try:
            prob.solve(solver=SOLVER, ignore_dpp=True)
            first = len(calls)
            A.value = np.eye(n) * 2.0
            b.value = np.ones(n) * 0.5
            prob.solve(solver=SOLVER, ignore_dpp=True)
            second = len(calls)
        finally:
            Dcp2Cone.apply = orig_apply

        # First solve runs the full chain (Dcp2Cone called). Second solve hits
        # the cached param_prog and skips Dcp2Cone entirely.
        self.assertEqual(first, 1)
        self.assertEqual(second, 1)

    def test_toggling_ignore_dpp_rebuilds_chain(self) -> None:
        """The cache key includes ignore_dpp, so flipping it between solves must
        rebuild the chain (diffengine vs DPP) and still produce matching answers.
        """
        n = 3
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        A.value = np.eye(n)
        b.value = np.ones(n)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(A @ x - b)), [x >= 0, x <= 1])

        # Solve #1: ignore_dpp=True selects the diffengine chain.
        prob.solve(solver=SOLVER, ignore_dpp=True)
        x_ignore = x.value.copy()
        self.assertTrue(_has_diffengine(prob._cache.solving_chain))

        # Solve #2: ignore_dpp=False must invalidate the cache and rebuild a DPP
        # chain (no diffengine, no EvalParams).
        prob.solve(solver=SOLVER)
        x_dpp = x.value.copy()
        self.assertFalse(_has_diffengine(prob._cache.solving_chain))
        self.assertFalse(_has_eval_params(prob._cache.solving_chain))
        self.assertItemsAlmostEqual(x_ignore, x_dpp, places=4)

        # Solve #3: back to ignore_dpp=True must rebuild the diffengine chain.
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertTrue(_has_diffengine(prob._cache.solving_chain))
        self.assertItemsAlmostEqual(x.value, x_dpp, places=4)


class TestIgnoreDppReformulation(BaseTest):
    def test_nonaffine_constraint_becomes_diffengine_after_canon(self) -> None:
        """Eligibility must be judged on the canonicalized problem, not the raw
        one. `abs(x) <= 1` is non-affine as written but becomes affine cone
        constraints after Dcp2Cone, so the diffengine backend should be selected.
        """
        n = 3
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        A.value = np.eye(n)
        b.value = np.ones(n)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum(A @ x - b)), [cp.abs(x) <= 1])

        chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        # The raw constraint arg (abs(x)) is not affine; only after canonicalization
        # is the diffengine path valid. The old raw-problem check rejected this.
        self.assertTrue(_has_diffengine(chain))
        self.assertFalse(_has_eval_params(chain))

        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertItemsAlmostEqual(x.value, [-1.0, -1.0, -1.0], places=4)


class TestIgnoreDppPSD(BaseTest):
    def test_psd_problem_uses_diffengine(self) -> None:
        """A PSD constraint with an affine objective is diffengine-able: after
        CvxAttr2Constr lowers the `symmetric` attribute the constraint args are
        affine, and the diffengine handles the PSD cone directly."""
        n = 3
        X = cp.Variable((n, n), symmetric=True)
        a = cp.Parameter()
        a.value = 1.0
        prob = cp.Problem(cp.Minimize(cp.trace(X)),
                          [X >> a * np.eye(n)])

        chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        self.assertTrue(_has_diffengine(chain))
        self.assertFalse(_has_eval_params(chain))

        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertAlmostEqual(prob.value, 3.0, places=4)


class TestIgnoreDppQuadObjective(BaseTest):
    """Quadratic objectives are handled on the diffengine path: Dcp2Cone leaves a
    SymbolicQuadForm, which the diff engine lowers to equivalent atoms and recovers
    P from as the autodiff Hessian. No fallback to EvalParams."""

    def test_quadratic_objective_uses_diffengine(self) -> None:
        n = 3
        a = cp.Parameter(n)
        a.value = np.ones(n)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - a)), [x >= 0])

        chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        self.assertTrue(_has_diffengine(chain))
        self.assertFalse(_has_eval_params(chain))

        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertItemsAlmostEqual(x.value, a.value, places=4)

    def test_sum_squares_param_matches_baseline(self) -> None:
        """A parametric least-squares objective: two solves with different
        parameter values must match the DPP baseline."""
        rng = np.random.default_rng(0)
        n = 5
        A = cp.Parameter((n, n))
        b = cp.Parameter(n)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= -1, x <= 1])
        base = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= -1, x <= 1])

        for _ in range(2):
            A.value = rng.standard_normal((n, n))
            b.value = rng.standard_normal(n)
            prob.solve(solver=SOLVER, ignore_dpp=True)
            x_de = x.value.copy()
            base.solve(solver=SOLVER)
            self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_quad_form_parametric_P_matches_baseline(self) -> None:
        """quad_form(x, P) with a parametric (param-affine) PSD P. The diff engine
        rejects a parametric P natively, so the SymbolicQuadForm is lowered to the
        multiply/sum form; the result must match the baseline."""
        rng = np.random.default_rng(1)
        n = 4
        M = rng.standard_normal((n, n))
        P = cp.Parameter((n, n), PSD=True)
        P.value = M @ M.T + np.eye(n)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + cp.sum(x)), [x >= -1, x <= 1])
        base = cp.Problem(cp.Minimize(cp.quad_form(x, P) + cp.sum(x)), [x >= -1, x <= 1])

        chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        self.assertTrue(_has_diffengine(chain))

        prob.solve(solver=SOLVER, ignore_dpp=True)
        x_de = x.value.copy()
        base.solve(solver=SOLVER)
        self.assertItemsAlmostEqual(x_de, x.value, places=4)

    def test_no_parameter_quadratic_objective(self) -> None:
        """A no-parameter quadratic objective (introduces an auxiliary variable
        for the compound argument) must solve correctly via the diffengine."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertItemsAlmostEqual(x.value, [1.0, 1.0, 1.0], places=4)
        self.assertAlmostEqual(prob.value, 0.0, places=4)

    def test_matrix_variable_quadratic_objective(self) -> None:
        """A quadratic objective over a *matrix* variable. The SymbolicQuadForm's
        P is sized for the flattened variable, so the lowering must vec the 2-D
        leaf before forming P @ x (else P @ x is e.g. (16,16) @ (4,4))."""
        n = 4
        X = cp.Variable((n, n))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X - 2 * np.eye(n))),
                          [cp.trace(X) == n])
        base = cp.Problem(cp.Minimize(cp.sum_squares(X - 2 * np.eye(n))),
                          [cp.trace(X) == n])

        chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        self.assertTrue(_has_diffengine(chain))

        prob.solve(solver=SOLVER, ignore_dpp=True)
        X_de = X.value.copy()
        base.solve(solver=SOLVER)
        self.assertItemsAlmostEqual(X_de, X.value, places=4)
        self.assertAlmostEqual(prob.value, base.value, places=4)
