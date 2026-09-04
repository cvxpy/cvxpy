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
from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, PowConeND, SvecPSD
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone.exact import ExactCone2Cone
from cvxpy.reductions.cone2cone.extract_identity_cones import ExtractIdentityCones
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.psd_utils import TriangleKind
from cvxpy.utilities.solver_context import SolverInfo


class TestExtractIdentityCones(BaseTest):
    CONTEXT = SolverInfo(
        supported_constraints={NonNeg, SOC, SvecPSD, ExpCone, PowCone3D, PowConeND},
        psd_triangle_kind=TriangleKind.UPPER, psd_sqrt2_scaling=True,
        x_cone_kinds={'nonneg', 'soc', 'psd_triangle', 'exp', 'power', 'gen_power'},
    )

    @classmethod
    def stuff(cls, problem):
        chain = Chain(reductions=[
            Dcp2Cone(solver_context=cls.CONTEXT), CvxAttr2Constr(reduce_bounds=True),
            ExactCone2Cone(target_cones={PSD}, solver_context=cls.CONTEXT),
            ConeMatrixStuffing(canon_backend=cp.SCIPY_CANON_BACKEND),
        ])
        return chain.apply(problem)[0]

    def test_cone_families(self):
        """Verify sub-cone order, metadata, and row removal without a solver."""
        x = cp.Variable((3, 2))
        t = cp.Variable(2)
        X = cp.Variable((2, 2, 2), symmetric=True)
        alpha = np.array([[0.3, 0.5], [0.4, 0.3], [0.3, 0.2]])
        cases = [
            (NonNeg(x), 'nonneg', [6], [{}]),
            (SOC(t, x), 'soc', [4, 4], [{}, {}]),
            (X >> 0, 'psd_triangle', [3, 3], [{'psd_k': 2}] * 2),
            (ExpCone(x[0], x[1], x[2]), 'exp', [3, 3], [{}, {}]),
            (PowCone3D(x[0], x[1], x[2], [0.3, 0.6]), 'power', [3, 3],
             [{'alpha': 0.3}, {'alpha': 0.6}]),
            (PowConeND(x, t, alpha), 'gen_power', [4, 4],
             [{'alphas': a.tolist(), 'dim2': 1} for a in alpha.T]),
            (PowConeND(x.T, t, alpha.T, axis=1), 'gen_power', [4, 4],
             [{'alphas': a.tolist(), 'dim2': 1} for a in alpha.T]),
        ]
        reduction = ExtractIdentityCones(self.CONTEXT)
        for constraint, kind, sizes, extras in cases:
            with self.subTest(kind=kind, constraint=constraint):
                original = self.stuff(cp.Problem(cp.Minimize(0), [constraint]))
                original = ConicSolver.format_constraints(original, [0, 1, 2])
                result, inverse = reduction.apply(original)
                self.assertIsNone(inverse)
                self.assertEqual([c.kind for c in result.x_cones], [kind] * len(sizes))
                self.assertEqual([len(c.indices) for c in result.x_cones], sizes)
                self.assertEqual([c.extras for c in result.x_cones], extras)
                self.assertEqual([c.constr_id for c in result.x_cones],
                                 [original.constraints[0].id] * len(sizes))
                self.assertEqual(result.constr_size, 0)
                _, _, A, b = original.apply_parameters()
                indices = np.concatenate([c.indices for c in result.x_cones])
                self.assertEqual(len(set(indices)), sum(sizes))
                expected = np.ones(sum(sizes))
                if kind == 'psd_triangle':
                    expected[1::3] = np.sqrt(2)
                np.testing.assert_allclose(A[:, indices].toarray(), np.diag(expected))
                np.testing.assert_array_equal(b, 0)
                self.assertEqual(result.apply_parameters()[2].shape, (0, original.x.size))

    def test_reject_nonidentity_blocks(self):
        x = cp.Variable(4)
        a = cp.Parameter(4, value=np.ones(4))
        offset = cp.Parameter(4, value=np.zeros(4))
        expressions = [2*x, -x, x+1, cp.multiply(a, x), x+offset,
                       cp.hstack([x[0], x[0], x[2], x[3]]),
                       cp.hstack([x[0], x[1]+x[2], 0, x[3]])]
        for expr in expressions:
            with self.subTest(expr=expr):
                original = self.stuff(cp.Problem(cp.Minimize(0), [expr >= 0]))
                result, _ = ExtractIdentityCones(self.CONTEXT).apply(original)
                self.assertEqual(result.x_cones, [])
                self.assertEqual(result.constr_size, 4)

    def test_disjointness_and_repeated_apply(self):
        x = cp.Variable(3)
        constraints = [NonNeg(x), SOC(x[0], x[1:])]
        original = self.stuff(cp.Problem(cp.Minimize(0), constraints))
        reduction = ExtractIdentityCones(self.CONTEXT)
        result, _ = reduction.apply(original)
        self.assertEqual([c.kind for c in result.x_cones], ['nonneg'])
        self.assertEqual([c.id for c in result.constraints], [constraints[1].id])
        again, _ = reduction.apply(result)
        self.assertIs(again, result)
        self.assertEqual(len(again.x_cones), 1)

    def test_opt_in_and_integer_variables(self):
        x = cp.Variable(2, integer=True)
        original = self.stuff(cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0]))
        result, inverse = ExtractIdentityCones().apply(original)
        self.assertIs(result, original)
        self.assertIsNone(inverse)
        result, _ = ExtractIdentityCones(self.CONTEXT).apply(original)
        self.assertEqual([c.kind for c in result.x_cones], ['nonneg'])
        self.assertEqual(result.x.integer_idx, original.x.integer_idx)
        self.assertIs(ExtractIdentityCones().invert(result, None), result)

    def test_parameter_updates_and_kept_rows(self):
        """Extraction keeps the parameter tensor valid after values change."""
        x = cp.Variable(3)
        a = cp.Parameter(3, value=np.ones(3))
        b = cp.Parameter(value=1.)
        constraints = [NonNeg(x), a @ x == b, 2*x + b >= 0]
        problem = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
        original = self.stuff(problem)
        original = ConicSolver.format_constraints(original, [0, 1, 2])
        result, _ = ExtractIdentityCones(self.CONTEXT).apply(original)
        self.assertEqual([c.kind for c in result.x_cones], ['nonneg'])
        kept = [0, 4, 5, 6]  # equality, then the nonidentity NonNeg block
        for a.value, b.value in [(np.ones(3), 1.), (np.array([2., -1., 0.5]), 3.)]:
            _, _, A, offset = original.apply_parameters()
            _, _, new_A, new_offset = result.apply_parameters()
            np.testing.assert_allclose(new_A.toarray(), A[kept].toarray())
            np.testing.assert_allclose(new_offset, offset[kept])
            # Reconstruct extracted constraints and compare with the original solve.
            y = cp.Variable(result.x.size)
            q, d, _, _ = result.apply_parameters()
            reconstructed = cp.Problem(cp.Minimize(q @ y + d),
                                       [new_A[:1] @ y + new_offset[:1] == 0,
                                        new_A[1:] @ y + new_offset[1:] >= 0,
                                        y[result.x_cones[0].indices] >= 0])
            expected = problem.solve(solver=cp.CLARABEL)
            self.assertAlmostEqual(reconstructed.solve(solver=cp.CLARABEL), expected)

    def test_extend_existing_cones(self):
        x = cp.Variable(3)
        t = cp.Variable()
        y = cp.Variable()
        original = self.stuff(cp.Problem(cp.Minimize(t + y), [SOC(t, x), y >= 0]))
        first, _ = ExtractIdentityCones(SolverInfo(x_cone_kinds={'nonneg'})).apply(original)
        result, _ = ExtractIdentityCones(self.CONTEXT).apply(first)
        self.assertEqual([c.kind for c in result.x_cones], ['nonneg', 'soc'])
        self.assertEqual(result.constr_size, 0)
