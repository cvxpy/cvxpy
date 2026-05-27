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
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


def _has_diffengine(chain):
    """True iff a `use_diffengine=True` ConeMatrixStuffing is in the chain."""
    return any(isinstance(r, ConeMatrixStuffing) and r.use_diffengine
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


class TestIgnoreDppFallback(BaseTest):
    def test_psd_problem_falls_back_to_eval_params(self) -> None:
        """A problem DIFFENGINE can't accept (PSD constraint) must silently
        fall back to the EvalParams path and still solve correctly."""
        n = 3
        X = cp.Variable((n, n), symmetric=True)
        a = cp.Parameter()
        a.value = 1.0
        prob = cp.Problem(cp.Minimize(cp.trace(X)),
                          [X >> a * np.eye(n)])

        chain = prob._construct_chain(solver=SOLVER, ignore_dpp=True)
        # Silent fallback: DIFFENGINE rejected, EvalParams in chain instead.
        self.assertFalse(_has_diffengine(chain))
        self.assertTrue(_has_eval_params(chain))

        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertAlmostEqual(prob.value, 3.0, places=4)
