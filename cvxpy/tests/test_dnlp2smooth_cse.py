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
from cvxpy.expressions import cvxtypes
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth import dnlp2smooth as dnlp2smooth_mod
from cvxpy.reductions.dnlp2smooth.dnlp2smooth import Dnlp2Smooth
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize


def _aux_variables(new_prob, originals):
    """Return Variables in ``new_prob`` that are not in ``originals``."""
    orig_ids = {v.id for v in originals}
    return [v for v in new_prob.variables() if v.id not in orig_ids]


class TestDnlp2SmoothCSE(BaseTest):
    """Verify that Dnlp2Smooth deduplicates structurally identical subtrees."""

    def test_huber_shared_obj_and_constraint(self) -> None:
        # huber_canon mints two aux Variables (n, s) plus an abs aux. Reusing
        # huber(x) in the objective and in a constraint must canonicalize once.
        x = cp.Variable(3)
        x.value = np.zeros(3)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(x))),
                          [cp.sum(cp.huber(x)) <= 5])

        new_prob, _ = Dnlp2Smooth().apply(prob)

        with_cse = _aux_variables(new_prob, [x])
        # Without CSE we would emit two copies of every huber aux (n, s, plus
        # the abs epigraph variable). With CSE the second occurrence is a hit.
        # Sanity check the count by canonicalizing the unshared form.
        y = cp.Variable(3)
        y.value = np.zeros(3)
        ref = cp.Problem(cp.Minimize(cp.sum(cp.huber(y))), [y >= -1])
        ref_prob, _ = Dnlp2Smooth().apply(ref)
        single_huber_aux = len(_aux_variables(ref_prob, [y]))

        self.assertEqual(len(with_cse), single_huber_aux)

    def test_pnorm_shared_in_two_constraints(self) -> None:
        # pnorm canonicalization mints a chain of aux variables; reusing
        # pnorm(x, 2) in two constraints must canonicalize once.
        x = cp.Variable(4)
        x.value = np.zeros(4)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.pnorm(x, 2) <= 3, cp.pnorm(x, 2) <= 4, x >= -5],
        )
        new_prob, _ = Dnlp2Smooth().apply(prob)
        with_cse = _aux_variables(new_prob, [x])

        # Baseline: a problem with a single pnorm occurrence.
        y = cp.Variable(4)
        y.value = np.zeros(4)
        ref = cp.Problem(cp.Minimize(cp.sum(y)),
                         [cp.pnorm(y, 2) <= 3, y >= -5])
        ref_prob, _ = Dnlp2Smooth().apply(ref)
        single_pnorm_aux = len(_aux_variables(ref_prob, [y]))

        self.assertEqual(len(with_cse), single_pnorm_aux)

    def test_distinct_subtrees_not_merged(self) -> None:
        # huber(x) and huber(-x) must not share aux variables.
        x = cp.Variable(2)
        x.value = np.zeros(2)
        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.huber(x)) + cp.sum(cp.huber(-x))),
            [x >= -1],
        )

        new_prob, _ = Dnlp2Smooth().apply(prob)
        with_cse = _aux_variables(new_prob, [x])

        # And the single-huber baseline:
        y = cp.Variable(2)
        y.value = np.zeros(2)
        ref = cp.Problem(cp.Minimize(cp.sum(cp.huber(y))), [y >= -1])
        ref_prob, _ = Dnlp2Smooth().apply(ref)
        single_huber_aux = len(_aux_variables(ref_prob, [y]))

        # Two structurally distinct huber subtrees produce twice as many aux
        # variables as one, because CSE refuses to merge them.
        self.assertEqual(len(with_cse), 2 * single_huber_aux)

    def test_parameter_subtree_dedup(self) -> None:
        # Parameter leaves key on .id, so reusing the same parameter inside two
        # structurally identical subtrees still deduplicates.
        p = cp.Parameter(2, nonneg=True)
        p.value = np.array([1.0, 2.0])
        x = cp.Variable(2)
        x.value = np.zeros(2)
        prob = cp.Problem(
            cp.Minimize(cp.sum(cp.huber(cp.multiply(p, x)))),
            [cp.sum(cp.huber(cp.multiply(p, x))) <= 10, x >= -1],
        )

        new_prob, _ = Dnlp2Smooth().apply(prob)
        with_cse = _aux_variables(new_prob, [x])

        # Baseline with a single occurrence:
        q = cp.Parameter(2, nonneg=True)
        q.value = np.array([1.0, 2.0])
        y = cp.Variable(2)
        y.value = np.zeros(2)
        ref = cp.Problem(cp.Minimize(cp.sum(cp.huber(cp.multiply(q, y)))),
                         [y >= -1])
        ref_prob, _ = Dnlp2Smooth().apply(ref)
        single_aux = len(_aux_variables(ref_prob, [y]))

        self.assertEqual(len(with_cse), single_aux)

    def test_constraint_id_preserved_under_dedup(self) -> None:
        # User constraints are excluded from CSE so their IDs flow through
        # inverse_data unchanged, even when their bodies share subtrees.
        x = cp.Variable(2)
        x.value = np.zeros(2)
        c1 = cp.sum(cp.huber(x)) <= 5
        c2 = cp.sum(cp.huber(x)) <= 8
        prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(x))), [c1, c2])

        reduction = Dnlp2Smooth()
        _, inverse_data = reduction.apply(prob)

        self.assertIn(c1.id, inverse_data.cons_id_map)
        self.assertIn(c2.id, inverse_data.cons_id_map)
        self.assertNotEqual(
            inverse_data.cons_id_map[c1.id],
            inverse_data.cons_id_map[c2.id],
        )

    def test_cache_cleared_between_applies(self) -> None:
        # The per-apply cache must not leak Variables across calls, so two
        # apply()s on equivalent problems mint fresh aux Variables.
        def build_problem():
            x = cp.Variable(2)
            x.value = np.zeros(2)
            return x, cp.Problem(
                cp.Minimize(cp.sum(cp.huber(x))),
                [cp.sum(cp.huber(x)) <= 5],
            )

        reduction = Dnlp2Smooth()

        x1, prob1 = build_problem()
        new1, _ = reduction.apply(prob1)
        aux1 = {v.id for v in _aux_variables(new1, [x1])}

        x2, prob2 = build_problem()
        new2, _ = reduction.apply(prob2)
        aux2 = {v.id for v in _aux_variables(new2, [x2])}

        # Fresh problem -> fresh Variables. No id overlap.
        self.assertEqual(aux1 & aux2, set())

    def test_shared_constraint_walk_is_correct(self) -> None:
        # Sanity: after canonicalization, every Variable referenced by the
        # canonicalized objective is also defined by the canonicalized
        # constraint set. Catches accidental dangling aux Variables that
        # could result from a bad cache hit.
        x = cp.Variable(3)
        x.value = np.zeros(3)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.huber(x))),
                          [cp.sum(cp.huber(x)) <= 4, x >= -2])

        new_prob, _ = Dnlp2Smooth().apply(prob)

        objective_vars = {v.id for v in new_prob.objective.variables()
                          if isinstance(v, Variable)}
        constraint_vars = set()
        for c in new_prob.constraints:
            constraint_vars.update(v.id for v in c.variables())
        # Every aux var in the objective must also appear in some constraint.
        self.assertTrue(objective_vars - {x.id} <= constraint_vars)

    def test_partial_optimize_excluded_from_cse_cache(self) -> None:
        x = cp.Variable()
        y = cp.Variable()
        y.value = 0
        inner = cp.Problem(cp.Minimize(cp.square(y - x)), [y >= 0])
        partial = partial_optimize(inner, opt_vars=[y])
        prob = cp.Problem(cp.Minimize(partial + cp.square(x)))

        seen_types = []
        real_expr_key = dnlp2smooth_mod.expr_key

        def spy_expr_key(expr, key_cache):
            seen_types.append(type(expr))
            return real_expr_key(expr, key_cache)

        with patch.object(dnlp2smooth_mod, "expr_key", side_effect=spy_expr_key):
            Dnlp2Smooth().apply(prob)

        self.assertNotIn(cvxtypes.partial_problem(), seen_types)

    def test_structural_key_cache_threaded_through_apply(self) -> None:
        # A single per-apply key cache keeps deep expression trees from
        # rebuilding structural keys for every suffix subtree.
        x = cp.Variable()
        x.value = 0
        expr = x
        for _ in range(20):
            expr = expr + 1
        prob = cp.Problem(cp.Minimize(expr), [expr <= 25])

        seen_key_caches = []
        real_expr_key = dnlp2smooth_mod.expr_key

        def spy_expr_key(expr, key_cache):
            seen_key_caches.append(key_cache)
            key = real_expr_key(expr, key_cache)
            self.assertIsInstance(key, int)
            return key

        reduction = Dnlp2Smooth()
        with patch.object(dnlp2smooth_mod, "expr_key", side_effect=spy_expr_key):
            reduction.apply(prob)

        self.assertTrue(seen_key_caches)
        self.assertNotIn(None, seen_key_caches)
        self.assertEqual(len({id(key_cache) for key_cache in seen_key_caches}), 1)
