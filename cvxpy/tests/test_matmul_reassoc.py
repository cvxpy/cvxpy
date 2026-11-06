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


class TestMatmulReassoc(BaseTest):

    def test_rewrite_preserves_value(self) -> None:
        rng = np.random.default_rng(0)
        A, C = rng.standard_normal((3, 4)), rng.standard_normal((4, 4))
        X = cp.Variable((4, 3), value=rng.standard_normal((4, 3)))
        z = cp.Variable((1, 1), value=[[3.0]])
        for expr in [A @ X @ np.ones((3, 2)), (X + C @ X) @ np.ones(3), (-X) @ np.ones(3),
                     X.T @ C @ np.ones(4), (2 * z.T) @ np.array([[7.0]])]:
            rewritten = _normalize_matmul(expr)
            self.assertIsNot(rewritten, expr)
            self.assertItemsAlmostEqual(rewritten.value, expr.value)

    def test_left_as_written(self) -> None:
        """1-D collapses and parametric tails must not be rewritten."""
        p = cp.Parameter((3, 2), value=np.ones((3, 2)))
        for expr in [(cp.Variable((2, 3)) @ np.ones(3)) @ np.ones(2),
                     (np.ones((3, 4)) @ cp.Variable(4)) @ np.ones(3),
                     (np.ones((3, 4)) @ cp.Variable((4, 3))) @ p]:
            self.assertIs(_normalize_matmul(expr), expr)

    def test_in_solving_chain(self) -> None:
        X = cp.Variable((4, 3))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(np.ones((3, 4)) @ X @ np.eye(3, 2) - 1)
                                      + cp.sum_squares(X)))
        prob.solve(solver=cp.CLARABEL)
        self.assertTrue(any(isinstance(r, MatmulReassoc)
                            for r in prob._cache.solving_chain.reductions))
        value = prob.value
        prob.solve(solver=cp.CLARABEL, canon_backend=s.DIFFENGINE_CANON_BACKEND)
        self.assertAlmostEqual(prob.value, value)
