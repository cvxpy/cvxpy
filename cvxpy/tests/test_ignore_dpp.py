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

import os
import unittest
import warnings
from unittest import mock

import numpy as np
import pytest

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.test_diffengine_backend import MISSING

pytestmark = pytest.mark.skipif(
    bool(MISSING),
    reason="DIFFENGINE backend requires sparsediffpy >= 0.6.0 "
           f"(missing: {', '.join(MISSING)})",
)

SOLVER = cp.CLARABEL


class TestIgnoreDppBehavior(BaseTest):
    """ignore_dpp / non-DPP solves keep parameters symbolic on the DIFFENGINE
    backend; values must refresh across solves and unsound canonicalizations
    must not happen."""

    def test_parametric_divisor_resolves(self) -> None:
        """A parametric divisor is re-evaluated by the engine on each solve."""
        p = cp.Parameter(pos=True)
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x / p - 1.0)))
        for val in (2.0, 5.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val, places=4)

        pv = cp.Parameter(3, pos=True)
        xv = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(xv / pv - 1.0)))
        for val in (np.array([1.0, 2.0, 4.0]), np.array([3.0, 0.5, 1.5])):
            pv.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertItemsAlmostEqual(xv.value, val, places=4)

    def test_param_constant_in_concave_position_sound(self) -> None:
        """A parametric-constant composite on the 'wrong' side of an
        inequality must not be epigraph-relaxed: x <= power(t, 2) would
        relax to x <= s, s >= t**2 (vacuous). It folds to a CallbackParam
        leaf before canonicalization and refreshes on each solve."""
        t = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(-x), [x <= cp.power(t, 2)])
        for val in (2.0, 3.0):
            t.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val ** 2, places=4)

    def test_unsupported_variable_free_atom_evaluates(self) -> None:
        """An atom with no symbolic converter over parameters only (floor,
        as emitted by DQCP bisection) folds to a CallbackParam leaf whose
        value refreshes per solve."""
        p = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - cp.floor(p))))
        for val in (2.7, -1.2):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(x.value, np.floor(val), places=4)

    def test_symbolic_quad_matrix_refreshes(self) -> None:
        """quad_over_lin(x, p) keeps its quadratic matrix symbolic (I/p fed
        to the engine as a composite param_source); re-solves must serve
        fresh values."""
        x = cp.Variable()
        p = cp.Parameter()
        prob = cp.Problem(cp.Minimize(cp.quad_over_lin(x, p) + x))
        for val in (1.0, 1000.0, 1.0):
            p.value = val
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(x.value, -val / 2.0, places=3)

    @unittest.skipUnless(INSTALLED_MI_SOLVERS, "no mixed-integer solver installed")
    def test_mixed_integer_ignore_dpp(self) -> None:
        x = cp.Variable(3, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0.5, x <= 3.7])
        prob.solve(ignore_dpp=True)
        self.assertEqual(prob.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(x.value, np.ones(3), places=4)


class TestIgnoreDppSelection(BaseTest):
    """The ignore_dpp / non-DPP path force-selects the DIFFENGINE backend
    over any user-supplied backend, with an N-D EvalParams fallback."""

    def _stuffing_backend(self, chain) -> str:
        stuffing = [r for r in chain.reductions
                    if isinstance(r, ConeMatrixStuffing)][0]
        return stuffing.canon_backend

    def test_ignore_dpp_defaults_to_diffengine(self) -> None:
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))
        _, chain, _ = prob.get_problem_data(SOLVER, ignore_dpp=True)
        self.assertEqual(self._stuffing_backend(chain), s.DIFFENGINE_CANON_BACKEND)
        prob.solve(solver=SOLVER, ignore_dpp=True)
        self.assertAlmostEqual(x.value, 2.0)

    def test_explicit_backend_with_params_raises(self) -> None:
        """A non-DIFFENGINE backend cannot serve the symbolic ignore_dpp path
        for parametric problems; parameter-free problems honor the request."""
        p = cp.Parameter()
        p.value = 2.0
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.square(x - p)))

        for backend in (s.CPP_CANON_BACKEND, s.SCIPY_CANON_BACKEND):
            with self.assertRaisesRegex(ValueError, "keeps parameters symbolic"):
                prob.get_problem_data(SOLVER, ignore_dpp=True,
                                      canon_backend=backend)

        # Parameter-free: ignore_dpp is a no-op, explicit backend honored.
        # (Fresh problem per backend: the problem-data cache does not key
        # on canon_backend.)
        for backend in (s.CPP_CANON_BACKEND, s.SCIPY_CANON_BACKEND):
            y = cp.Variable(3)
            prob_free = cp.Problem(cp.Minimize(cp.sum_squares(y - 1)), [y >= 0])
            _, chain, _ = prob_free.get_problem_data(SOLVER, ignore_dpp=True,
                                                     canon_backend=backend)
            self.assertEqual(self._stuffing_backend(chain), backend)
            prob_free.solve(solver=SOLVER, ignore_dpp=True, canon_backend=backend)
            self.assertAlmostEqual(prob_free.value, 0.0)

    def test_env_var_does_not_override_ignore_dpp(self) -> None:
        with mock.patch.dict(
                os.environ,
                {"CVXPY_DEFAULT_CANON_BACKEND": s.SCIPY_CANON_BACKEND}):
            p = cp.Parameter()
            p.value = 3.0
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(cp.square(x - p)))
            _, chain, _ = prob.get_problem_data(SOLVER, ignore_dpp=True)
            self.assertEqual(self._stuffing_backend(chain),
                             s.DIFFENGINE_CANON_BACKEND)
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertAlmostEqual(x.value, 3.0)

    def test_nd_problem_falls_back_like_cpp(self) -> None:
        """>2-D problems cannot use the (2-D) diff engine: the ignore_dpp path
        bakes parameters (EvalParams) like the CPP -> SCIPY N-D fallback, and
        the baked values must still refresh across solves."""
        p = cp.Parameter()
        x = cp.Variable((2, 2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - p)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # N-D -> SCIPY fallback warning
            p.value = 1.0
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertNotEqual(
                self._stuffing_backend(prob._cache.solving_chain),
                s.DIFFENGINE_CANON_BACKEND)
            reduction_names = [type(r).__name__
                               for r in prob._cache.solving_chain.reductions]
            self.assertIn('EvalParams', reduction_names)
            self.assertItemsAlmostEqual(x.value, np.full((2, 2, 2), 1.0), places=4)

            p.value = -2.0
            prob.solve(solver=SOLVER, ignore_dpp=True)
            self.assertItemsAlmostEqual(x.value, np.full((2, 2, 2), -2.0), places=4)
