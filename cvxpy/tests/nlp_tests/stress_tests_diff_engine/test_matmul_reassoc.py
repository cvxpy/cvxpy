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
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import _normalize_matmul
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


def _cases(rng):
    """(expression, X) pairs covering each rewrite rule in _apply_constant_right."""
    A = rng.standard_normal((3, 4))
    C_sq = rng.standard_normal((4, 4))
    C2 = rng.standard_normal((3, 2))
    c_vec = rng.standard_normal(3)
    c_vec4 = rng.standard_normal(4)
    X = cp.Variable((4, 3))
    return X, [
        A @ X @ C2,               # (A @ X) @ C2: matmul recursion
        (X + C_sq @ X) @ c_vec,   # AddExpression with a vector tail
        (-X) @ c_vec,             # NegExpression push-through
        X.T @ C_sq @ c_vec4,      # (E @ C) @ c: constants fold together
    ]


class TestNormalizeMatmul:
    """_normalize_matmul rewrites a matmul chain ending in a plain constant.
    The rewrite must fire (so the tests observe it, not just the answer) and
    must be value-preserving.
    """

    def test_const_tail_rewritten_and_value_preserving(self):
        rng = np.random.default_rng(0)
        X, cases = _cases(rng)
        X.value = rng.standard_normal((4, 3))
        for expr in cases:
            rewritten = _normalize_matmul(expr)
            assert rewritten is not expr, f"rewrite did not fire for {expr}"
            assert rewritten.shape == expr.shape
            np.testing.assert_allclose(
                np.asarray(rewritten.value), np.asarray(expr.value), atol=1e-10)

    def test_1d_collapse_left_as_written(self):
        """A chain contracting through a 1-D operand must not be touched:
        (a @ b) @ c != a @ (b @ c) across numpy's 1-D collapse."""
        rng = np.random.default_rng(0)
        X = cp.Variable((2, 3))
        expr = (X @ np.array([1.0, 2.0, 3.0])) @ np.array([1.0, 1.0])
        assert _normalize_matmul(expr) is expr

        y = cp.Variable(4)
        expr2 = (rng.standard_normal((3, 4)) @ y) @ rng.standard_normal(3)
        assert _normalize_matmul(expr2) is expr2

    def test_parametric_tail_not_folded(self):
        """A parametric tail must stay symbolic -- folding would freeze the
        current value into a Constant."""
        p = cp.Parameter((3, 2))
        p.value = np.ones((3, 2))
        expr = (np.ones((3, 4)) @ cp.Variable((4, 3))) @ p
        assert _normalize_matmul(expr) is expr


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestReassociatedChainDerivatives:
    """The engine must differentiate the rewritten tree correctly."""

    def test_derivatives_match_on_reassociated_chains(self):
        rng = np.random.default_rng(0)
        X, cases = _cases(rng)
        for expr in cases:
            target = rng.standard_normal(expr.shape)
            X.value = np.zeros((4, 3))
            prob = cp.Problem(cp.Minimize(
                cp.sum_squares(expr - target) + cp.sum_squares(X)))
            DerivativeChecker(prob).run_and_assert()
