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
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.atoms.quad_form import QuadForm, SymbolicQuadForm
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.dcp2cone import dcp2cone as dcp2cone_mod
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.subexpr_cache import StructuralKeyCache, expr_key
from cvxpy.tests.base_test import BaseTest


def _find_atoms(expr, atom_cls):
    """Return all atoms of ``atom_cls`` reachable from ``expr.args``."""
    found = []
    stack = [expr]
    while stack:
        e = stack.pop()
        if isinstance(e, atom_cls):
            found.append(e)
        stack.extend(getattr(e, "args", []))
    return found


class TestDcp2ConeCSE(BaseTest):
    """Verify that Dcp2Cone deduplicates structurally identical subtrees."""

    def test_scalar_norm1_dedup(self) -> None:
        # Reported case: norm1(x) appears in objective and constraint.
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.norm1(x)), [cp.norm1(x) <= 1])
        new_prob, _ = Dcp2Cone().apply(prob)

        # Exactly one auxiliary variable beyond the original x.
        new_vars = new_prob.variables()
        self.assertEqual(len(new_vars), 2)
        aux_vars = [v for v in new_vars if v is not x]
        self.assertEqual(len(aux_vars), 1)
        self.assertEqual(aux_vars[0].shape, ())

        # Exactly one pair of abs-epigraph inequalities: t >= x and t >= -x.
        ineqs = [c for c in new_prob.constraints if isinstance(c, Inequality)]
        epigraph_ineqs = [c for c in ineqs
                          if any(v is x for v in c.variables())]
        self.assertEqual(len(epigraph_ineqs), 2)

    def test_vector_norm1_dedup(self) -> None:
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.norm1(x)), [cp.norm1(x) <= 5])
        new_prob, _ = Dcp2Cone().apply(prob)

        aux_vars = [v for v in new_prob.variables() if v is not x]
        self.assertEqual(len(aux_vars), 1)
        self.assertEqual(aux_vars[0].shape, (3,))

    def test_distinct_subtrees_not_merged(self) -> None:
        # norm1(x) and norm1(-x) must NOT share an epigraph variable.
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.norm1(x) + cp.norm1(-x)), [x >= -1])
        new_prob, _ = Dcp2Cone().apply(prob)

        aux_vars = [v for v in new_prob.variables() if v is not x]
        # One epigraph variable for norm1(x), one for norm1(-x).
        self.assertEqual(len(aux_vars), 2)

    def test_solve_matches_unduplicated(self) -> None:
        # Solving a problem with a duplicated norm1 should agree with an
        # equivalent problem that uses a single shared expression.
        x = cp.Variable(3)
        prob_dup = cp.Problem(
            cp.Minimize(cp.norm1(x) + cp.sum(x)),
            [cp.norm1(x) <= 4, cp.sum(x) >= 1],
        )
        prob_dup.solve(solver=cp.CLARABEL)

        y = cp.Variable(3)
        shared = cp.norm1(y)
        prob_shared = cp.Problem(
            cp.Minimize(shared + cp.sum(y)),
            [shared <= 4, cp.sum(y) >= 1],
        )
        prob_shared.solve(solver=cp.CLARABEL)

        self.assertEqual(prob_dup.status, cp.OPTIMAL)
        self.assertAlmostEqual(prob_dup.value, prob_shared.value, places=5)
        self.assertItemsAlmostEqual(x.value, y.value, places=5)

    def test_shared_quad_form_solves_correctly(self) -> None:
        # A shared QuadForm reused in 0.5*qf + 0.5*qf must solve to the same
        # answer as the unshared problem: the downstream coefficient extractor
        # needs to keep the two occurrences as distinct rows even when CSE
        # makes them share a canonical SymbolicQuadForm.
        np.random.seed(0)
        A = np.random.randn(5, 5)
        z = np.random.randn(5)
        P = A.T @ A
        q = -2 * P @ z
        w = cp.Variable(5)
        qf = QuadForm(w, P)
        prob = cp.Problem(cp.Minimize(0.5 * qf + 0.5 * qf + q.T @ w))
        prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(w.value, z, places=4)

    def test_quad_obj_shared_subtree_dedup(self) -> None:
        # With quad_obj=True, a quad-eligible subtree (here quad_over_lin)
        # appearing twice in the objective should produce a single shared
        # SymbolicQuadForm in the canonicalized expression.
        x = cp.Variable(3)
        qol = cp.quad_over_lin(x, 1)
        prob = cp.Problem(cp.Minimize(qol + qol + cp.sum(x)))

        new_prob, _ = Dcp2Cone(quad_obj=True).apply(prob)

        sqfs = _find_atoms(new_prob.objective.expr, SymbolicQuadForm)
        self.assertEqual(len(sqfs), 2)
        self.assertTrue(all(s is sqfs[0] for s in sqfs))

        # And the problem still solves to the same answer as the explicit
        # 2*qol formulation.
        y = cp.Variable(3)
        ref = cp.Problem(cp.Minimize(2 * cp.quad_over_lin(y, 1) + cp.sum(y)))
        prob.solve(solver=cp.CLARABEL)
        ref.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, ref.value, places=5)
        self.assertItemsAlmostEqual(x.value, y.value, places=5)

    def test_quad_obj_cross_context_not_merged(self) -> None:
        # The same quad_over_lin subtree used in the objective and in a
        # constraint must canonicalize differently: the objective takes the
        # quad branch (SymbolicQuadForm) and the constraint takes the cone
        # branch (SOC). The affine_above component of the cache key keeps
        # those results separate even though the structural key matches.
        x = cp.Variable(3)
        qol = cp.quad_over_lin(x, 1)
        prob = cp.Problem(cp.Minimize(qol + cp.sum(x)), [qol <= 5])

        new_prob, _ = Dcp2Cone(quad_obj=True).apply(prob)

        self.assertEqual(len(_find_atoms(new_prob.objective.expr,
                                         SymbolicQuadForm)), 1)
        self.assertEqual(
            sum(1 for c in new_prob.constraints if isinstance(c, SOC)), 1
        )

        prob.solve(solver=cp.CLARABEL)
        self.assertEqual(prob.status, cp.OPTIMAL)

    def test_shared_ndarray_constant_dedup(self) -> None:
        # A large float64 ndarray passed to two separate cp.Constant(...)
        # calls produces two distinct Constant wrappers that share the same
        # underlying ndarray (the ndarray interface stores by reference for
        # float64). _constant_key's id-of-underlying branch merges them.
        arr = np.random.default_rng(0).standard_normal(100)
        x = cp.Variable(100)
        c1 = cp.Constant(arr)
        c2 = cp.Constant(arr)
        # Two structurally identical norm1(multiply(c_i, x)) subtrees.
        prob = cp.Problem(
            cp.Minimize(cp.norm1(cp.multiply(c1, x))),
            [cp.norm1(cp.multiply(c2, x)) <= 5, x >= -1],
        )
        new_prob, _ = Dcp2Cone().apply(prob)

        aux_vars = [v for v in new_prob.variables() if v is not x]
        self.assertEqual(len(aux_vars), 1)
        self.assertEqual(aux_vars[0].shape, (100,))

    def test_sparse_constant_key_uses_sparse_contents(self) -> None:
        rows = np.array([0, 1])
        cols = np.array([1, 0])
        data = np.array([1.0, 2.0])
        same_data = np.array([2.0, 1.0])
        different_data = np.array([1.0, 3.0])
        duplicate_data = np.array([0.25, 0.75, 2.0])
        duplicate_rows = np.array([0, 0, 1])
        duplicate_cols = np.array([1, 1, 0])

        coo = sp.coo_array((data, (rows, cols)), shape=(2, 2))
        csr_same = sp.csr_array((same_data, (cols, rows)), shape=(2, 2))
        coo_same_with_duplicates = sp.coo_array(
            (duplicate_data, (duplicate_rows, duplicate_cols)), shape=(2, 2))
        coo_different = sp.coo_array((different_data, (rows, cols)), shape=(2, 2))
        const = cp.Constant(coo)
        const_same = cp.Constant(csr_same)
        const_same_with_duplicates = cp.Constant(coo_same_with_duplicates)
        const_different = cp.Constant(coo_different)

        key_cache = StructuralKeyCache()
        key = expr_key(const, key_cache)
        same_key = expr_key(const_same, key_cache)
        same_duplicate_key = expr_key(const_same_with_duplicates, key_cache)
        different_key = expr_key(const_different, key_cache)

        self.assertEqual(key, same_key)
        self.assertEqual(key, same_duplicate_key)
        self.assertNotEqual(key, different_key)

    def test_parameter_subtree_dedup(self) -> None:
        # Parameter leaves are keyed by id; the same parameter reused in two
        # identical subtrees should also deduplicate.
        p = cp.Parameter(2, nonneg=True)
        p.value = np.array([1.0, 2.0])
        x = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(cp.norm1(cp.multiply(p, x))),
            [cp.norm1(cp.multiply(p, x)) <= 10, x >= 0],
        )
        new_prob, _ = Dcp2Cone().apply(prob)

        aux_vars = [v for v in new_prob.variables() if v is not x]
        # Only the single epigraph variable for the shared norm1 subtree.
        self.assertEqual(len(aux_vars), 1)
        self.assertEqual(aux_vars[0].shape, (2,))

    def test_structural_key_cache_threaded_through_apply(self) -> None:
        # The structural key builder sees every node during canonicalization.
        # A single per-apply key cache prevents deep expression chains from
        # rebuilding each suffix key at every node.
        x = cp.Variable()
        expr = x
        for _ in range(20):
            expr = expr + 1
        prob = cp.Problem(cp.Minimize(expr), [expr <= 25])

        seen_key_caches = []
        real_expr_key = dcp2cone_mod.expr_key

        def spy_expr_key(expr, key_cache):
            seen_key_caches.append(key_cache)
            key = real_expr_key(expr, key_cache)
            self.assertIsInstance(key, int)
            return key

        reduction = Dcp2Cone()
        with patch.object(dcp2cone_mod, "expr_key", side_effect=spy_expr_key):
            reduction.apply(prob)

        self.assertTrue(seen_key_caches)
        self.assertNotIn(None, seen_key_caches)
        self.assertEqual(len({id(key_cache) for key_cache in seen_key_caches}), 1)
