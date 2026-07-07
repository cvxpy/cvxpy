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
from cvxpy.expressions.constants.callback_param import CallbackParam
from cvxpy.reductions.fold_callback_params import CallbackParamFold, _fold_expr
from cvxpy.tests.base_test import BaseTest


class TestCallbackParamFold(BaseTest):
    """The fold replaces exactly the non-affine variable-free parametric
    subtrees, at their maximal (outermost) node, with refreshable
    CallbackParam leaves."""

    def test_non_affine_subtree_folds_and_refreshes(self) -> None:
        t = cp.Parameter()
        folded = _fold_expr(cp.power(t, 2))
        self.assertIsInstance(folded, CallbackParam)
        for val in (2.0, -3.0):
            t.value = val
            self.assertAlmostEqual(float(folded.value), val ** 2)

    def test_maximal_subtree_single_leaf(self) -> None:
        """The whole variable-free composite folds as ONE leaf, not one per
        nested non-affine node."""
        t = cp.Parameter(pos=True)
        expr = cp.exp(cp.square(t) + 1.0)
        folded = _fold_expr(expr)
        self.assertIsInstance(folded, CallbackParam)
        t.value = 2.0
        self.assertAlmostEqual(float(folded.value), np.exp(5.0))

    def test_affine_and_leaf_subtrees_stay_symbolic(self) -> None:
        p = cp.Parameter(2)
        self.assertIs(_fold_expr(p), p)
        affine = 2.0 * p + np.ones(2)
        self.assertIs(_fold_expr(affine), affine)

    def test_variable_branches_untouched_identity_preserved(self) -> None:
        """Folding descends past variables; an expression with nothing to
        fold is returned as the SAME object (cache-key stability)."""
        t = cp.Parameter()
        x = cp.Variable()
        expr = x + cp.square(t)
        folded = _fold_expr(expr)
        self.assertIsNot(folded, expr)
        self.assertIsInstance(folded.args[1], CallbackParam)

        no_params = x + cp.square(cp.Constant(2.0))
        self.assertIs(_fold_expr(no_params), no_params)

    def test_sign_is_propagated(self) -> None:
        t = cp.Parameter()
        folded = _fold_expr(cp.square(t))
        self.assertTrue(folded.is_nonneg())
        folded_neg = _fold_expr(-cp.square(t))
        self.assertTrue(folded_neg.is_nonpos())

    def test_problem_apply_rebuilds_constraints(self) -> None:
        t = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(-x), [x <= cp.power(t, 2)])
        new_prob, _ = CallbackParamFold().apply(prob)
        (con,) = new_prob.constraints
        params = new_prob.parameters()
        self.assertEqual(len(params), 1)
        self.assertIsInstance(params[0], CallbackParam)
        t.value = 3.0
        self.assertAlmostEqual(float(con.args[1].value), 9.0)

    def test_epigraph_soundness_end_to_end(self) -> None:
        """x <= power(t, 2) must not be epigraph-relaxed (vacuous) on the
        ignore_dpp path; the folded value refreshes between solves."""
        t = cp.Parameter()
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(-x), [x <= cp.power(t, 2)])
        for val in (2.0, 3.0):
            t.value = val
            prob.solve(solver=cp.CLARABEL, ignore_dpp=True)
            self.assertEqual(prob.status, cp.OPTIMAL)
            self.assertAlmostEqual(x.value, val ** 2, places=4)
