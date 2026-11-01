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
from cvxpy.reductions.algebra import MatmulReassoc
from cvxpy.reductions.algebra.matmul_reassoc import _normalize_matmul
from cvxpy.tests.base_test import BaseTest


def reassoc_cases(rng):
    """(X, expressions) covering each rewrite rule of _apply_constant_right."""
    A = rng.standard_normal((3, 4))
    C_sq = rng.standard_normal((4, 4))
    X = cp.Variable((4, 3))
    return X, [
        A @ X @ rng.standard_normal((3, 2)),          # (A @ X) @ C
        (X + C_sq @ X) @ rng.standard_normal(3),      # sum with a vector tail
        (-X) @ rng.standard_normal(3),                # negation
        X.T @ C_sq @ rng.standard_normal(4),          # (E @ C) @ c folds C @ c
    ]


class TestMatmulReassoc(BaseTest):

    def test_const_tail_rewritten_and_value_preserving(self) -> None:
        rng = np.random.default_rng(0)
        X, cases = reassoc_cases(rng)
        X.value = rng.standard_normal((4, 3))
        for expr in cases:
            rewritten = _normalize_matmul(expr)
            self.assertIsNot(rewritten, expr)
            self.assertEqual(rewritten.shape, expr.shape)
            self.assertItemsAlmostEqual(rewritten.value, expr.value, places=10)

    def test_1d_collapse_left_as_written(self) -> None:
        """(a @ b) @ c != a @ (b @ c) across numpy's 1-D collapse."""
        X = cp.Variable((2, 3))
        expr = (X @ np.array([1.0, 2.0, 3.0])) @ np.array([1.0, 1.0])
        self.assertIs(_normalize_matmul(expr), expr)
        expr = (np.ones((3, 4)) @ cp.Variable(4)) @ np.ones(3)
        self.assertIs(_normalize_matmul(expr), expr)

    def test_parametric_tail_not_folded(self) -> None:
        p = cp.Parameter((3, 2), value=np.ones((3, 2)))
        expr = (np.ones((3, 4)) @ cp.Variable((4, 3))) @ p
        self.assertIs(_normalize_matmul(expr), expr)

    def test_elementwise_multiply_not_treated_as_matmul(self) -> None:
        """multiply subclasses MulExpression; 2 * z.T must not be split as a matmul."""
        z = cp.Variable((1, 1), value=[[3.0]])
        expr = (2 * z.T) @ np.array([[7.0]])
        self.assertItemsAlmostEqual(_normalize_matmul(expr).value, expr.value)

    def test_in_solving_chain(self) -> None:
        """Every canon backend runs the reduction and solves to the same value."""
        rng = np.random.default_rng(0)
        A, C = rng.standard_normal((3, 4)), rng.standard_normal((3, 2))
        X = cp.Variable((4, 3))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ X @ C - 1) + cp.sum_squares(X)))
        values = []
        for backend in [None, s.SCIPY_CANON_BACKEND, s.DIFFENGINE_CANON_BACKEND]:
            prob.solve(solver=cp.CLARABEL, canon_backend=backend)
            self.assertTrue(any(isinstance(r, MatmulReassoc)
                                for r in prob._cache.solving_chain.reductions))
            values.append(prob.value)
        self.assertItemsAlmostEqual(values, [values[0]] * len(values), places=5)
