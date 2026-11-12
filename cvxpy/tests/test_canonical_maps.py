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
import warnings

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.cvxcore.python.canonInterface import reduce_problem_data_tensor
from cvxpy.error import DPPError
from cvxpy.reductions.canonical_maps import (
    AffineSparseMatrix,
    Exp,
    ParameterTransformKind,
    Recovery,
    StuffedTensor,
    UnsupportedRecovery,
    trace_scaled_selection,
    transform_source_value,
)
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS, SOLVER_MAP_QP
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.tests.base_test import BaseTest

CONIC = [s for s in (cp.CLARABEL, cp.SCS) if s in INSTALLED_SOLVERS]
QP = [s for s in (cp.OSQP,) if s in INSTALLED_SOLVERS]
# Interfaces taking the split form A x = b, h - G x in K.
SPLIT = [s for s in (cp.ECOS, cp.QOCO) if s in INSTALLED_SOLVERS]
ECOS = [s for s in SPLIT if s == cp.ECOS]
# Data keys under which each converted interface's apply() stores its inputs.
DATA_KEYS = {
    cp.CLARABEL: {'P': 'P', 'q': 'c', 'A': 'A', 'b': 'b'},
    cp.SCS: {'P': 'P', 'c': 'c', 'A': 'A', 'b': 'b'},
    cp.DIFFCP: {'c': 'c', 'A': 'A', 'b': 'b'},
}


def _dense(value):
    return value.toarray() if sp.issparse(value) else np.asarray(value)


def _solver_vectors(solver, raw):
    """The primal and dual vectors the recovery maps read."""
    if solver == cp.CLARABEL:
        return np.array(raw.x), np.array(raw.z)
    if solver == cp.OSQP:
        return raw.x, raw.y
    if solver == cp.ECOS:
        return raw['x'], np.concatenate([raw['y'], raw['z']])
    if solver == cp.QOCO:
        return np.array(raw.x), np.concatenate([np.array(raw.y), np.array(raw.z)])
    return raw['x'], raw['y']


def _stored_value(var):
    """A variable's value in the form its recovery map produces."""
    if var.attributes['sparsity']:
        return np.asarray(var._value)
    if var.attributes['diag']:
        return var.value.diagonal()
    return np.asarray(var.value)


def _source_values(problem):
    return {p.id: transform_source_value(p) for p in problem.parameters()}


def _reference_inputs(param_prog, solver):
    """Solver inputs from ``ParamConeProg.apply_parameters``, keyed as the maps are.

    Converted interfaces no longer call ``apply_parameters``, so this is an
    independent reference for the evaluated maps.
    """
    if param_prog.P is None:
        P, (c, d, A, b) = None, param_prog.apply_parameters()
    else:
        P, c, d, A, b = param_prog.apply_parameters(quad_obj=True)
    n = c.size
    P = np.triu(_dense(P)) if P is not None else np.zeros((n, n))
    ref = {'P': P, 'd': np.atleast_1d(d)}
    A, b = -_dense(A), np.asarray(b)
    if solver == cp.OSQP:
        # l <= A x <= u stacks equalities (l = u) above inequalities (l = -inf).
        p = param_prog.cone_dims.zero
        sign = np.where(np.arange(b.size) < p, -1.0, 1.0)
        ref.update(q=c, A=sign[:, None] * A, u=sign * b,
                   l=np.concatenate([-b[:p], np.full(b.size - p, -np.inf)]))
    elif solver in SPLIT or solver == cp.ECOS_BB:
        p = param_prog.cone_dims.zero
        ref.update(c=c, A=A[:p], b=b[:p], G=A[p:], h=b[p:])
    else:
        ref.update({'q' if solver == cp.CLARABEL else 'c': c, 'A': A, 'b': b})
    return ref


class TestCanonicalMaps(BaseTest):

    def setUp(self) -> None:
        warnings.simplefilter("ignore")

    def tearDown(self) -> None:
        warnings.resetwarnings()

    def assert_recovery_matches_invert(self, problem, solver, gp=False):
        """Recovering from the raw solver output must agree with ``invert``."""
        canon = problem.get_canonical_maps(solver, gp=gp)
        data, chain, inverse_data = problem.get_problem_data(solver, gp=gp, solver_opts={})
        raw = chain.solve_via_data(problem, data)
        problem.unpack_results(raw, chain, inverse_data)
        recovered = canon.recover(*_solver_vectors(solver, raw))
        variables = {v.id: v for v in problem.variables()}
        duals = {dv.id: dv for c in problem.constraints for dv in c.dual_variables}
        for info in canon.primal_vars:
            self.assertItemsAlmostEqual(recovered[info.var_id],
                                        _stored_value(variables[info.var_id]), places=6)
        for info in canon.dual_vars:
            expected = np.ravel(duals[info.dual_var_id].value, order='F')
            self.assertItemsAlmostEqual(np.ravel(recovered[info.dual_var_id], order='F'),
                                        expected, places=6)
        return canon

    def assert_inputs_match_problem_data(self, problem, solver, gp=False):
        """Evaluated maps agree with ``apply_parameters`` and the solve path's data."""
        canon = problem.get_canonical_maps(solver, gp=gp)
        data, _, inverse_data = problem.get_problem_data(solver, gp=gp)
        values = canon.evaluate(canon.prepare_param_vec(_source_values(problem)))
        reference = _reference_inputs(data[cp.settings.PARAM_PROB], solver)
        self.assertEqual(set(values) - {'P'}, set(reference) - {'P'})
        for key, value in values.items():
            np.testing.assert_allclose(_dense(value), reference[key], atol=1e-8, err_msg=key)
        for key, data_key in DATA_KEYS.get(solver, {}).items():
            if key in values:
                self.assertItemsAlmostEqual(_dense(values[key]), _dense(data[data_key]))
        self.assertAlmostEqual(values['d'][0], inverse_data[-1]['offset'])
        return canon, values

    def test_example(self) -> None:
        """The example from the design note."""
        p = cp.Parameter(2, value=np.array([1.0, -2.0]))
        x = cp.Variable(2)
        problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x) + p @ x), [x >= 0])
        canon = problem.get_canonical_maps(solver=cp.CLARABEL)
        M, v = canon.maps["q"]
        self.assertItemsAlmostEqual(M @ p.value.ravel(order="F") + v, p.value)
        self.assertEqual(canon.total_param_size, 2)
        self.assertEqual(canon.param_transforms, [])
        structure = canon.structures["P"]
        self.assertEqual(structure.format, "csc")
        self.assertEqual(structure.shape, (2, 2))

    def test_recovery(self) -> None:
        """Primal and dual recovery across cones, attributes and reductions."""
        rng = np.random.default_rng(0)
        problems = {}

        x = cp.Variable(3)
        p = cp.Parameter(3, value=np.ones(3))
        problems['qp'] = (cp.Problem(
            cp.Maximize(p @ x - cp.sum_squares(x)),
            [x[0] + x[1] == 1, x <= 2, cp.NonPos(x[2] - 1), x >= -3]), CONIC + QP + SPLIT)

        X, t, u = cp.Variable((3, 4)), cp.Variable(4), cp.Variable(3)
        problems['soc'] = (cp.Problem(
            cp.Minimize(cp.sum(t) + cp.sum(u) + cp.sum(X)),
            [cp.SOC(t, X, axis=0), cp.SOC(u, X, axis=1), cp.SOC(t[0], X[:, 0]),
             X >= -1, cp.norm(X[0]) <= 3]), CONIC + SPLIT)

        S = cp.Variable((3, 3), PSD=True)
        Y = cp.Variable((3, 3), symmetric=True)
        C = rng.standard_normal((3, 3))
        problems['psd'] = (cp.Problem(
            cp.Minimize(cp.trace((C + C.T) @ S) + cp.trace(Y)),
            [cp.trace(S) == 1, Y >> S, Y[0, 1] == 0.3]), CONIC)

        B = cp.Variable((2, 3, 3))
        problems['batched_psd'] = (cp.Problem(
            cp.Minimize(cp.sum(B)), [cp.PSD(B), cp.sum(B[:, 0, 0]) == 1, B >= -2]), CONIC)

        z = cp.Variable(3)
        problems['exp'] = (cp.Problem(
            cp.Minimize(-cp.sum(cp.entr(z)) + cp.sum(z)),
            [cp.sum(z) == 1, cp.ExpCone(z[0], z[1] + 1, z[2] + 2)]), CONIC + ECOS)

        w, W = cp.Variable(3, pos=True), cp.Variable((3, 2))
        alpha = np.array([[0.2, 0.3], [0.3, 0.3], [0.5, 0.4]])
        problems['pow'] = (cp.Problem(
            cp.Maximize(cp.geo_mean(w)),
            [cp.sum(w) <= 3, cp.PowCone3D(w[0], w[1], w[2] - 0.5, 0.3),
             cp.PowConeND(W, cp.Variable(2), alpha, axis=0), W <= 1]), CONIC)

        a, b, xx = cp.Variable(2), cp.Variable(2), cp.Variable((3, 2))
        problems['rsoc'] = (cp.Problem(
            cp.Minimize(cp.sum(a + b)), [cp.RSOC(a, b, xx, axis=0), xx == 1]), CONIC + SPLIT)

        D = cp.Variable((3, 3), diag=True)
        Sp = cp.Variable((3, 3), sparsity=([0, 1, 2], [1, 2, 0]))
        N = cp.Variable(3, nonneg=True)
        problems['attributes'] = (cp.Problem(
            cp.Minimize(cp.sum_squares(D) + cp.sum_squares(Sp) + cp.sum(N)),
            [cp.sum(D) >= 1, cp.sum(Sp) == 2, N >= 0.5]), CONIC + QP + SPLIT)

        v, e = cp.Variable(2), cp.Variable(0)
        problems['zero_sized'] = (cp.Problem(
            cp.Minimize(cp.sum_squares(v - 1) + cp.sum(e)), [v >= 0, e >= 0]), CONIC + QP + SPLIT)

        xe, pe = cp.Variable(2), cp.Parameter(2, value=np.array([1.0, -2.0]))
        problems['equality_only'] = (cp.Problem(
            cp.Minimize(cp.sum(xe)), [xe == pe]), CONIC + QP + SPLIT)

        pc = cp.Parameter(2, complex=True, value=np.array([1 + 2j, 3 - 1j]))
        xc = cp.Variable(2)
        problems['complex_param'] = (cp.Problem(
            cp.Minimize(cp.sum_squares(xc) + cp.real(cp.conj(pc) @ xc)), [xc >= -5]), CONIC)

        for name, (problem, solvers) in problems.items():
            for solver in solvers:
                with self.subTest(problem=name, solver=solver):
                    self.assert_recovery_matches_invert(problem, solver)
                    self.assert_inputs_match_problem_data(problem, solver)

    def test_empty_blocks_are_none(self) -> None:
        """Empty blocks are passed as ``None``, as before canonical maps.

        ECOS gets ``None`` for an empty equality or cone block; QOCO only for
        an empty equality block.
        """
        x, p = cp.Variable(2), cp.Parameter(2, value=np.array([1.0, -2.0]))
        only_eq = cp.Problem(cp.Minimize(cp.sum(x)), [x == p])
        only_ineq = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
        for solver in SPLIT:
            with self.subTest(solver=solver):
                data, _, _ = only_eq.get_problem_data(solver)
                if solver == cp.ECOS:
                    self.assertIsNone(data['G'])
                    self.assertIsNone(data['h'])
                else:
                    self.assertEqual(data['G'].shape, (0, 2))
                self.assertEqual(data['A'].shape, (2, 2))
                data, _, _ = only_ineq.get_problem_data(solver)
                self.assertIsNone(data['A'])
                self.assertIsNone(data['b'])
                for problem in (only_eq, only_ineq):
                    problem.solve(solver=solver)
                    self.assertItemsAlmostEqual(x.value, p.value, places=5)

    @unittest.skipUnless(cp.ECOS_BB in INSTALLED_SOLVERS, 'ECOS_BB is not installed.')
    def test_ecos_bb(self) -> None:
        """ECOS_BB inherits ECOS's maps; the integer indices stay in its data."""
        x = cp.Variable(2, integer=True)
        p = cp.Parameter(2, value=np.array([1.5, 2.5]))
        problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p, x[0] + x[1] == 5])
        self.assert_inputs_match_problem_data(problem, cp.ECOS_BB)
        data, _, _ = problem.get_problem_data(cp.ECOS_BB)
        self.assertEqual(list(data[cp.settings.INT_IDX]), [0, 1])
        problem.solve(solver=cp.ECOS_BB)
        self.assertItemsAlmostEqual(x.value, [2, 3])
        p.value = np.array([0.5, 3.5])
        problem.solve(solver=cp.ECOS_BB)
        self.assertItemsAlmostEqual(x.value, [1, 4])

    @unittest.skipUnless(cp.CLARABEL in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
    def test_gp(self) -> None:
        x, y = cp.Variable(pos=True), cp.Variable(pos=True)
        a = cp.Parameter(pos=True, value=2.0)
        problem = cp.Problem(cp.Minimize(a * x * y + 1 / x), [x * y >= 1, y <= 3])
        canon = self.assert_recovery_matches_invert(problem, cp.CLARABEL, gp=True)
        self.assertIsInstance(canon.primal_recovery[x.id], Exp)
        self.assertEqual([t.kind for t in canon.param_transforms], [ParameterTransformKind.LOG])
        a.value = 3.5
        self.assert_inputs_match_problem_data(problem, cp.CLARABEL, gp=True)

    @unittest.skipUnless(cp.CLARABEL in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
    def test_psd_dual_is_scaled(self) -> None:
        """Unpacking a scaled svec dual needs a matrix, not a gather."""
        X = cp.Variable((2, 2), symmetric=True)
        con = X >> 0
        problem = cp.Problem(cp.Minimize(cp.trace(X)), [con, X[0, 0] == 1])
        canon = problem.get_canonical_maps(cp.CLARABEL, [X, con.dual_variables[0]])
        self.assertTrue(sp.issparse(canon.dual_recovery[con.dual_variables[0].id]))
        # The symmetric variable is expanded from its upper triangle by a gather.
        (info,) = canon.primal_vars
        self.assertTrue(info.is_symmetric)
        self.assertEqual(info.size, 4)
        self.assertEqual(np.unique(canon.primal_recovery[X.id]).size, 3)

    def test_parameter_preparation(self) -> None:
        """Transformed, packed parameters follow the DPP fast path."""
        rng = np.random.default_rng(1)
        x, y = cp.Variable((3, 3)), cp.Variable(3)
        Pd = cp.Parameter((3, 3), diag=True, value=np.diag([1., 2., 3.]))
        Ps = cp.Parameter((3, 3), symmetric=True, value=np.eye(3))
        Pp = cp.Parameter((3, 3), PSD=True, value=np.eye(3))
        Psp = cp.Parameter((3, 3), sparsity=([0, 2], [0, 1]))
        Psp.value_sparse = sp.coo_array(([2.0, 3.0], ([0, 2], [0, 1])), shape=(3, 3))
        q = cp.Parameter(3, value=np.ones(3))
        pc = cp.Parameter(2, complex=True, value=np.array([1 + 2j, 3 - 1j]))
        H = cp.Parameter((3, 3), hermitian=True, value=np.eye(3))
        Pcs = cp.Parameter((2, 2), complex=True, sparsity=([0, 1], [1, 0]))
        Pcs.value_sparse = sp.coo_array((np.array([1 + 1j, 2 - 1j]), ([0, 1], [1, 0])),
                                        shape=(2, 2))
        xc, Z, V = cp.Variable(2), cp.Variable((3, 3), symmetric=True), cp.Variable((2, 2))
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(x) + cp.trace(Pd @ x) + cp.sum(cp.multiply(Ps, x))
                        + cp.trace(Pp @ x) + cp.sum(Psp @ y) + q @ y
                        + cp.sum_squares(xc) + cp.real(cp.conj(pc) @ xc)
                        + cp.real(cp.trace(H @ Z)) + cp.sum_squares(V)
                        + cp.real(cp.sum(cp.multiply(Pcs, V)))),
            [y >= -1, x <= 2, xc >= -5, Z >> 0, cp.trace(Z) == 1])
        for solver in CONIC:
            canon = problem.get_canonical_maps(solver)
            kinds = {t.kind for t in canon.param_transforms}
            self.assertEqual(kinds, set(ParameterTransformKind) - {ParameterTransformKind.LOG})
            A = rng.standard_normal((3, 3))
            Pd.value = np.diag(rng.standard_normal(3))
            Ps.value = A + A.T
            Pp.value = A @ A.T
            Psp.value_sparse = sp.coo_array((rng.standard_normal(2), ([0, 2], [0, 1])),
                                            shape=(3, 3))
            q.value = rng.standard_normal(3)
            pc.value = rng.standard_normal(2) + 1j * rng.standard_normal(2)
            B = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
            H.value = B + B.conj().T
            Pcs.value_sparse = sp.coo_array(
                (rng.standard_normal(2) + 1j * rng.standard_normal(2), ([0, 1], [1, 0])),
                shape=(2, 2))
            data, _, _ = problem.get_problem_data(solver)
            self.assertItemsAlmostEqual(canon.prepare_param_vec(_source_values(problem)),
                                        data[cp.settings.PARAM_PROB].param_vec())
            self.assert_inputs_match_problem_data(problem, solver)

    @unittest.skipUnless(cp.OSQP in INSTALLED_SOLVERS, 'OSQP is not installed.')
    def test_osqp_inputs(self) -> None:
        """OSQP's l/u: unbounded rows are constant -inf offsets, not map columns."""
        x = cp.Variable(2)
        p = cp.Parameter(nonneg=True, value=1.0)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x) + x[0]),
                             [cp.sum(x) == p, x <= 2 * p])
        canon, _ = self.assert_inputs_match_problem_data(problem, cp.OSQP)
        M, v = canon.maps['l']
        self.assertEqual(M[1:].nnz, 0)
        self.assertTrue(np.all(np.isneginf(v[1:])))
        problem.solve(solver=cp.OSQP)
        self.assertItemsAlmostEqual(x.value, [0.25, 0.75], places=3)

    @unittest.skipUnless(cp.CLARABEL in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
    def test_structure_is_value_independent(self) -> None:
        """A parametric entry stays structural when its value is zero."""
        x = cp.Variable(2)
        p = cp.Parameter(value=0.0)
        problem = cp.Problem(cp.Minimize(cp.sum(x)), [p * x[0] + x[1] >= 1, x >= 0])
        canon = problem.get_canonical_maps(cp.CLARABEL)
        values = canon.evaluate(canon.prepare_param_vec(_source_values(problem)))
        self.assertEqual(values['A'].nnz, 4)
        # The selective-update rule: only inputs with a nonzero column change.
        updated = {name for name, (M, _) in canon.maps.items() if M.nnz}
        self.assertEqual(updated, {'A'})

    @unittest.skipUnless(cp.CLARABEL in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
    def test_variable_selection(self) -> None:
        """Duals are grouped per dual variable."""
        x, t = cp.Variable(3), cp.Variable()
        soc = cp.SOC(t, x)
        problem = cp.Problem(cp.Minimize(t), [soc, cp.sum(x) == 1])
        canon = problem.get_canonical_maps(cp.CLARABEL, [soc.dual_variables[1], x])
        self.assertEqual([info.var_id for info in canon.primal_vars], [x.id])
        (info,) = canon.dual_vars
        self.assertEqual((info.dual_var_id, info.constraint_id), (soc.dual_variables[1].id, soc.id))
        self.assertEqual((info.shape, info.cone_type), ((3,), 'SOC'))
        with self.assertRaises(ValueError):
            problem.get_canonical_maps(cp.CLARABEL, [cp.Variable()])

    @unittest.skipUnless(cp.CLARABEL in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
    def test_chained_transforms_re_solve(self) -> None:
        """Re-solving after Complex2Real -> CvxAttr2Constr transforms matches a fresh solve.

        The hermitian parameter's imaginary part passes through an
        intermediate ID without a ``Parameter``; the complex sparse parameter
        is split into sparse real and imaginary parameters, which are then
        gathered.
        """
        rng = np.random.default_rng(3)
        H = cp.Parameter((3, 3), hermitian=True)
        Pcs = cp.Parameter((2, 2), complex=True, sparsity=([0, 1], [1, 0]))
        V, Z = cp.Variable((2, 2)), cp.Variable((3, 3), symmetric=True)
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(V) + cp.real(cp.sum(cp.multiply(Pcs, V)))
                        + cp.real(cp.trace(H @ Z))),
            [Z >> 0, cp.trace(Z) == 1])
        for _ in range(3):
            B = rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
            H.value = B + B.conj().T
            Pcs.value_sparse = sp.coo_array(
                (rng.standard_normal(2) + 1j * rng.standard_normal(2), ([0, 1], [1, 0])),
                shape=(2, 2))
            problem.solve(solver=cp.CLARABEL)
            V_cached = V.value
            fresh = cp.Problem(problem.objective, problem.constraints)
            fresh.solve(solver=cp.CLARABEL)
            self.assertAlmostEqual(problem.value, fresh.value, places=5)
            self.assertItemsAlmostEqual(V_cached, V.value, places=5)

    def test_no_variables(self) -> None:
        p = cp.Parameter(value=1.0)
        with self.assertRaisesRegex(ValueError, "no variables"):
            cp.Problem(cp.Minimize(p)).get_canonical_maps(cp.CLARABEL)

    def test_unsupported(self) -> None:
        x = cp.Variable(2)
        p = cp.Parameter(value=1.0)
        if cp.SCIPY in INSTALLED_SOLVERS:
            problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
            with self.assertRaisesRegex(NotImplementedError, "SCIPY"):
                problem.get_canonical_maps(cp.SCIPY)
        problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p * p])
        with self.assertRaises(DPPError):
            problem.get_canonical_maps(cp.CLARABEL)
        z = cp.Variable(2, complex=True)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(z - 1j) + cp.sum(x)), [x >= 0])
        with self.assertRaisesRegex(NotImplementedError, "complex"):
            problem.get_canonical_maps(cp.CLARABEL)
        canon = problem.get_canonical_maps(cp.CLARABEL, [x])
        self.assertEqual(len(canon.primal_vars), 1)

    def test_recovery_degrades_instead_of_raising(self) -> None:
        """Unsupported compositions mark the output; they do not raise."""
        gather = Recovery('x', 4, indices=[0, 2])
        self.assertIsInstance(gather.with_exp().linear(np.eye(2)), UnsupportedRecovery)
        summed = Recovery('x', 4, matrix=np.array([[1.0, 1.0, 0, 0]]))
        self.assertIsInstance(summed.with_exp().finalize(), UnsupportedRecovery)
        self.assertIsInstance(gather.with_exp().finalize(), Exp)
        self.assertIsInstance(Recovery.concat([gather, UnsupportedRecovery("r")]),
                              UnsupportedRecovery)
        with self.assertRaises(ValueError):
            Recovery.concat([])

    def test_trace_scaled_selection(self) -> None:
        S = trace_scaled_selection(lambda v: np.array([2 * v[2], 0.0, -v[0]]), 3)
        np.testing.assert_array_equal(S.toarray(), [[0, 0, 2], [0, 0, 0], [-1, 0, 0]])
        # Sums of inputs, including ones whose scales cancel, are rejected.
        for fn in (lambda v: np.array([v[0] + v[2]]), lambda v: np.array([v[0] - v[1]]),
                   lambda v: np.array([v[0] + 2 * v[1]])):
            with self.assertRaises(NotImplementedError):
                trace_scaled_selection(fn, 3)

    def test_to_structure_rejects_duplicates(self) -> None:
        M = sp.csr_array((2, 1))
        dup = AffineSparseMatrix((2, 2), [1, 1], [0, 0], M, [1.0, 2.0])
        with self.assertRaises(ValueError):
            dup.to_structure()

    def test_stuffed_tensor_matches_legacy_reduction(self) -> None:
        """``StuffedTensor`` keeps the rows ``reduce_problem_data_tensor`` keeps.

        The tensor has explicit zeros, duplicates that cancel, and empty
        rows; both drop stored zeros before summing duplicates, so cancelled
        entries remain as explicit zeros in both.
        """
        n_var, n_constr, n_cols = 2, 3, 2
        rows = [0, 0, 1, 4, 4, 5, 7]
        cols = [0, 1, 0, 1, 1, 0, 1]
        data = [1.0, 0.0, 2.0, 1.0, -1.0, 3.0, 4.0]
        tensor = sp.csc_array((data, (rows, cols)), shape=((n_var + 1) * n_constr, n_cols))
        new = StuffedTensor(tensor)
        legacy, indices, indptr, _ = reduce_problem_data_tensor(tensor.copy(), n_var)
        np.testing.assert_array_equal(new.nonzero_rows % n_constr, indices)
        np.testing.assert_array_equal(
            np.bincount(new.nonzero_rows // n_constr, minlength=n_var + 1),
            np.diff(indptr))
        np.testing.assert_array_equal(new.reduced.toarray(), legacy.toarray())
        self.assertEqual(new.reduced.nnz, sp.csr_array(legacy).nnz)

    def test_qp_interfaces_pair_maps(self) -> None:
        """A QP interface exporting maps must also turn them into its data."""
        for solver in SOLVER_MAP_QP.values():
            cls = type(solver)
            if cls.get_canonical_maps is not Solver.get_canonical_maps:
                with self.subTest(solver=solver.name()):
                    self.assertIsNot(cls._apply_canonical_maps, QpSolver._apply_canonical_maps)

    @unittest.skipUnless(cp.SCS in INSTALLED_SOLVERS, 'SCS is not installed.')
    def test_maps_cached_with_compilation(self) -> None:
        x = cp.Variable(2)
        p = cp.Parameter(2, value=np.ones(2))
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [x >= 0])
        problem.solve(solver=cp.SCS)
        param_prog = problem._cache.param_prog
        maps = param_prog._canonical_maps_cache[cp.reductions.solvers.conic_solvers.scs_conif.SCS]
        p.value = np.array([2.0, -1.0])
        problem.solve(solver=cp.SCS)
        self.assertIs(problem._cache.param_prog, param_prog)
        self.assertIs(param_prog._canonical_maps_cache[type(problem._cache.solving_chain.solver)],
                      maps)
        self.assertItemsAlmostEqual(x.value, [2.0, 0.0], places=4)

    @unittest.skipUnless(cp.DIFFCP in INSTALLED_SOLVERS, 'DIFFCP is not installed.')
    def test_param_jacobian_is_map_transpose(self) -> None:
        """``apply_param_jac`` agrees with the transposed canonical maps."""
        rng = np.random.default_rng(2)
        x = cp.Variable(3)
        A = cp.Parameter((2, 3), value=rng.standard_normal((2, 3)))
        b = cp.Parameter(2, value=rng.standard_normal(2))
        c = cp.Parameter(3, value=rng.standard_normal(3))
        problem = cp.Problem(cp.Minimize(c @ x + cp.norm(x, 1)), [A @ x <= b, x >= -1])
        canon, values = self.assert_inputs_match_problem_data(problem, cp.DIFFCP)
        data, _, _ = problem.get_problem_data(cp.DIFFCP)
        dc = rng.standard_normal(values['c'].size)
        db = rng.standard_normal(values['b'].size)
        structure = canon.structures['A']
        dA_vals = rng.standard_normal(structure.indices.size)
        grad = canon.maps['c'][0].T @ dc + canon.maps['A'][0].T @ dA_vals \
            + canon.maps['b'][0].T @ db
        dA = structure.assemble(dA_vals)
        # ``data['A']`` is the negated stuffed matrix, as in Problem.backward.
        expected = data[cp.settings.PARAM_PROB].apply_param_jac(dc, -dA, db)
        for pid, col in canon.param_id_to_col.items():
            size = canon.param_id_to_size[pid]
            self.assertItemsAlmostEqual(grad[col:col + size],
                                        np.ravel(expected[pid], order='F'))
