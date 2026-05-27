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
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.nonpos import Inequality
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.tests.base_test import BaseTest


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
