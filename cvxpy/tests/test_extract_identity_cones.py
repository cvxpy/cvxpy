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

import unittest

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
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
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

    def test_unrestricted_psd(self):
        for shape in [(2, 2), (3, 3), (2, 3, 3)]:
            with self.subTest(shape=shape):
                X = cp.Variable(shape)
                original = self.stuff(cp.Problem(cp.Minimize(0), [X >> 0]))
                result, _ = ExtractIdentityCones(self.CONTEXT).apply(original)
                self.assertEqual(result.constr_size, 0)
                self.assertEqual(result.x.size, X.size)
                self.assertEqual(len(result.x_cones), X.size // shape[-1]**2)
                _, _, A, b = original.apply_parameters()
                x = np.arange(X.size, dtype=float)
                packed = []
                used = []
                for cone in result.x_cones:
                    n = cone.extras['psd_k']
                    self.assertEqual(len(cone.psd_pairs), n * (n - 1) // 2)
                    values = x[cone.indices].copy()
                    for i, j in cone.psd_pairs:
                        values[cone.indices.index(i)] = (x[i] + x[j]) / np.sqrt(2)
                    packed.extend(values)
                    used.extend(cone.indices + [j for _, j in cone.psd_pairs])
                self.assertEqual(sorted(used), list(range(X.size)))
                np.testing.assert_allclose(packed, A @ x + b)

    def test_unrestricted_psd_rejection(self):
        X = cp.Variable((2, 2))
        a = cp.Parameter((2, 2), value=np.ones((2, 2)))
        for expr in [X + np.eye(2), cp.multiply(a, X),
                     cp.bmat([[X[0, 0], X[0, 0]], [X[1, 0], X[1, 1]]])]:
            with self.subTest(expr=expr):
                original = self.stuff(cp.Problem(cp.Minimize(0), [expr >> 0]))
                result, _ = ExtractIdentityCones(self.CONTEXT).apply(original)
                self.assertEqual(result.x_cones, [])
        for attr in ['integer', 'lower_bounds', 'lb_tensor']:
            with self.subTest(attr=attr):
                original = self.stuff(cp.Problem(cp.Minimize(0), [X >> 0]))
                original = ConicSolver.format_constraints(original, [0, 1, 2])
                if attr == 'integer':
                    original.x.attributes['integer'] = [(0,)]
                elif attr == 'lb_tensor':
                    original.lb_tensor = original.q[:X.size]
                else:
                    setattr(original, attr, np.zeros(X.size))
                result, _ = ExtractIdentityCones(self.CONTEXT).apply(original)
                self.assertEqual(result.x_cones, [])

    def test_unrestricted_psd_overlap(self):
        X = cp.Variable((2, 2))
        reduction = ExtractIdentityCones(self.CONTEXT)
        # Either member of an off-diagonal pair must remain disjoint from
        # other direct cones, including cones extracted by a previous apply.
        for entry in [X[0, 1], X[1, 0]]:
            original = self.stuff(cp.Problem(cp.Minimize(0), [entry >= 0, X >> 0]))
            result, _ = reduction.apply(original)
            self.assertEqual([c.kind for c in result.x_cones], ['nonneg'])
        original = self.stuff(cp.Problem(cp.Minimize(0), [X >> 0, X.T >> 0]))
        result, _ = reduction.apply(original)
        self.assertEqual(len(result.x_cones), 1)
        self.assertIs(reduction.apply(result)[0], result)


@unittest.skipUnless('MOREAU' in INSTALLED_SOLVERS, 'MOREAU is not installed.')
class TestMoreauUnrestrictedPSD(BaseTest):
    def test_skew_primal_and_dual(self):
        X = cp.Variable((2, 2))
        target = np.array([[1., 4.], [0., 1.]])
        constraint = X >> 0
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(X) - 2 * cp.sum(cp.multiply(target, X))),
            [constraint],
        )
        data, _, _ = problem.get_problem_data(cp.MOREAU)
        self.assertEqual(data[ConicSolver.DIMS].psd, [])
        self.assertEqual(len(data['x_cones'][0].psd_pairs), 1)
        self.assertAlmostEqual(problem.solve(solver=cp.MOREAU), -17., places=4)
        np.testing.assert_allclose(X.value, [[1.5, 3.5], [-0.5, 1.5]], atol=1e-4)
        np.testing.assert_allclose(constraint.dual_value, [[1., -1.], [-1., 1.]], atol=1e-4)

    def test_batched_dpp_with_skew_constraints(self):
        X = cp.Variable((2, 2, 2), bounds=[-1.5, 1.5])
        skew = cp.Parameter(value=1.)
        target = cp.Parameter(X.shape, value=np.arange(8.).reshape(X.shape))
        coefficients = np.arange(1., 9.).reshape(X.shape)
        constraints = [X >> 0, X[0, 0, 1] - X[0, 1, 0] == skew,
                       X[1, 0, 1] + 2 * X[1, 1, 0] == 1]
        problem = cp.Problem(cp.Minimize(cp.sum(cp.multiply(coefficients, cp.square(X)))
                                        - 2 * cp.sum(cp.multiply(target, X))), constraints)
        reference = cp.Problem(problem.objective, constraints)
        cached = None
        for skew.value in [1., -2.]:
            target.value = target.value + 0.5
            value = problem.solve(solver=cp.MOREAU, enforce_dpp=True)
            primal = X.value.copy()
            duals = [np.array(c.dual_value, copy=True) for c in constraints]
            if cached is not None:
                self.assertIs(problem._cache.param_prog, cached)
            cached = problem._cache.param_prog
            data, _, _ = problem.get_problem_data(cp.MOREAU)
            self.assertEqual(data[ConicSolver.DIMS].psd, [])
            self.assertEqual(len(data['x_cones']), 2)
            self.assertAlmostEqual(reference.solve(solver=cp.CLARABEL), value, places=4)
            np.testing.assert_allclose(X.value, primal, atol=2e-4)
            for c, dual in zip(constraints, duals):
                np.testing.assert_allclose(c.dual_value, dual, atol=2e-4)
