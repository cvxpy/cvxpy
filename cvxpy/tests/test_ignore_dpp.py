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
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_diffengine_backend import MISSING

pytestmark = pytest.mark.skipif(
    bool(MISSING),
    reason="DIFFENGINE backend requires sparsediffpy >= 0.6.1 "
           f"(missing: {', '.join(MISSING)})",
)

SOLVER = cp.CLARABEL
DIFFENGINE = s.DIFFENGINE_CANON_BACKEND


def _stuffing_backend(chain) -> str | None:
    stuffing = [r for r in chain.reductions
                if isinstance(r, ConeMatrixStuffing)][0]
    return stuffing.canon_backend


class TestIgnoreDppSelection(BaseTest):
    """The ignore_dpp / non-DPP branch defaults to the DIFFENGINE backend when
    the stuffed problem is parameter-free, and falls back silently otherwise."""

    def test_ignore_dpp_defaults_to_diffengine(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
        # EvalParams chains are never cached; the program is rebuilt per solve.
        self.assertIsNone(prob._cache.param_prog)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)

    def test_parametric_ignore_dpp_bakes_then_defaults(self) -> None:
        """Parametric problems stay in scope: EvalParams bakes the values, so
        the stuffed problem is parameter-free and tracks the parameter across
        (fully recompiled) re-solves."""
        p = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p * p)))  # not DPP
        for val in (1.0, -3.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(_stuffing_backend(prob._cache.solving_chain),
                             DIFFENGINE)
            chain_types = [type(r) for r in prob._cache.solving_chain.reductions]
            self.assertIn(EvalParams, chain_types)
            self.assertAlmostEqual(x.value, val * val, places=5)

    def test_non_dpp_defaults_to_diffengine(self) -> None:
        """The non-DPP branch (without an explicit ignore_dpp) selects the
        backend the same way."""
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p * p)))
        with pytest.warns(UserWarning, match="not DPP"):
            prob.solve(solver=SOLVER)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)
        self.assertAlmostEqual(x.value, 4.0, places=5)

    def test_baked_parametric_data_matches_cpp(self) -> None:
        """Stuffed data on the default DIFFENGINE route equals an explicit CPP
        request at the same parameter values."""
        p = cp.Parameter(2)
        p.value = np.array([1.0, -2.0])
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)),
                          [cp.norm2(x) <= 3, cp.sum(x) == 0.5])
        data_de, _, _ = prob.get_problem_data(SOLVER, ignore_dpp=True)
        data_cpp, _, _ = prob.get_problem_data(
            SOLVER, ignore_dpp=True, canon_backend=s.CPP_CANON_BACKEND)
        for key in data_cpp:
            expected = data_cpp[key]
            if sp.issparse(expected):
                self.assertItemsAlmostEqual(
                    data_de[key].toarray(), expected.toarray(), places=10)
            elif isinstance(expected, np.ndarray):
                self.assertItemsAlmostEqual(data_de[key], expected, places=10)

    def test_nd_problem_falls_back(self) -> None:
        """N-D expressions fall back to the tensor backends silently."""
        x = cp.Variable((2, 2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)))
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain),
                            DIFFENGINE)
        self.assertAlmostEqual(prob.value, 0.0, places=6)

    def test_parametric_bounds_fall_back(self) -> None:
        """Parametric variable bounds survive EvalParams, so default selection
        must fall back to the tensor backends instead of tripping the
        backend's guards."""
        lb = cp.Parameter(2)
        lb.value = np.array([0.5, 0.25])
        x = cp.Variable(2, bounds=[lb, 10])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain),
                            DIFFENGINE)
        self.assertItemsAlmostEqual(x.value, lb.value, places=4)

    def test_parametric_pow_cone_alpha(self) -> None:
        """PowCone3D's alpha lives outside the constraint args and survives
        EvalParams; it must flow to the solver identically on the default
        DIFFENGINE route."""
        alpha = cp.Parameter()
        alpha.value = 0.4
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        cons = [cp.constraints.PowCone3D(x, y, z, alpha), x <= 2, y <= 3]
        prob = cp.Problem(cp.Maximize(z), cons)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertEqual(_stuffing_backend(prob._cache.solving_chain), DIFFENGINE)

        x2 = cp.Variable(pos=True)
        y2 = cp.Variable(pos=True)
        z2 = cp.Variable()
        base = cp.Problem(cp.Maximize(z2),
                          [cp.constraints.PowCone3D(x2, y2, z2, 0.4),
                           x2 <= 2, y2 <= 3])
        base.solve(solver=SOLVER, canon_backend=s.CPP_CANON_BACKEND,
                   ignore_dpp=True)
        self.assertAlmostEqual(prob.value, base.value, places=4)

    def test_explicit_backend_still_honored(self) -> None:
        """An explicit canon_backend on the ignore_dpp path wins over the
        default."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)))
        _, chain, _ = prob.get_problem_data(
            SOLVER, ignore_dpp=True, canon_backend=s.SCIPY_CANON_BACKEND)
        self.assertEqual(_stuffing_backend(chain), s.SCIPY_CANON_BACKEND)

    def test_dpp_path_unaffected(self) -> None:
        """DPP solves without ignore_dpp keep the tensor pipeline."""
        p = cp.Parameter()
        p.value = 1.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)))
        prob.solve(solver=SOLVER)
        self.assertNotEqual(_stuffing_backend(prob._cache.solving_chain),
                            DIFFENGINE)

    def test_perspective_solves_with_pinned_internal_chain(self) -> None:
        """The perspective canonicalizer's internal ignore_dpp chain is pinned
        to CPP; perspective problems keep solving correctly."""
        x = cp.Variable()
        t = cp.Variable(pos=True)
        persp = cp.perspective(cp.square(x), s=t)
        prob = cp.Problem(cp.Minimize(persp + t), [x >= 2, t <= 4])
        prob.solve(solver=SOLVER)
        self.assertEqual(prob.status, cp.OPTIMAL)
        # x^2/t + t minimized at t = min(x=2 -> 2), value 2^2/2+2 = 4 at t=2
        self.assertAlmostEqual(prob.value, 4.0, places=3)
