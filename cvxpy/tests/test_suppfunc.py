"""
Copyright 2019, the cvxpy developers.

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
import pytest

import cvxpy as cp
from cvxpy.constraints.psd import PSD, SvecPSD
from cvxpy.error import SolverError
from cvxpy.reductions.dcp2cone.canonicalizers.suppfunc_canon import (
    suppfunc_canon,
)
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.copt_conif import COPT
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (
    SCS,
    scs_psdvec_to_psdmat,
)
from cvxpy.reductions.solvers.conic_solvers.sdpa_conif import SDPA
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.suppfunc import (
    _coniclift,
    scs_cone_selectors,
    scs_coniclift,
)
from cvxpy.utilities.solver_context import SolverInfo


def _solver_context(solver: type[ConicSolver]) -> SolverInfo:
    """Build the context used by canonicalization without running the solver."""
    return SolverInfo(
        solver=solver().name(),
        supported_constraints=frozenset(solver.SUPPORTED_CONSTRAINTS),
        supports_bounds=solver.BOUNDED_VARIABLES,
        psd_triangle_kind=solver.PSD_TRIANGLE_KIND,
        psd_sqrt2_scaling=solver.PSD_SQRT2_SCALING,
    )


class TestSupportFunctions(BaseTest):
    """
    Test the implementation of support function atoms.

    Relevant source code includes:
        cvxpy.atoms.suppfunc
        cvxpy.transforms.suppfunc
        cvxpy.reductions.dcp2cone.canonicalizers.suppfunc_canon
    """

    def test_Rn(self) -> None:
        np.random.seed(0)
        n = 5
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [])
        a = np.random.randn(n,)
        y = cp.Variable(shape=(n,))
        cons = [sigma(y - a) <= 0]  # "<= num" for any num >= 0 is valid.
        objective = cp.Minimize(a @ y)
        prob = cp.Problem(objective, cons)
        prob.solve(solver='CLARABEL')
        actual = prob.value
        expected = np.dot(a, a)
        self.assertLessEqual(abs(actual - expected), 1e-6)
        actual = y.value
        expected = a
        self.assertLessEqual(np.linalg.norm(actual - expected, ord=2), 1e-6)
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-8)

    def test_vector1norm(self) -> None:
        n = 3
        np.random.seed(1)
        a = np.random.randn(n,)
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [cp.norm(x - a, 1) <= 1])
        y = np.random.randn(n,)
        y_var = cp.Variable(shape=(n,))
        prob = cp.Problem(cp.Minimize(sigma(y_var)), [y == y_var])
        prob.solve(solver='CLARABEL')
        actual = prob.value
        expected = a @ y + np.linalg.norm(y, ord=np.inf)
        self.assertLessEqual(abs(actual - expected), 1e-5)
        self.assertLessEqual(abs(prob.objective.expr.value - prob.value), 1e-5)

    def test_vector2norm(self) -> None:
        n = 3
        np.random.seed(1)
        a = np.random.randn(n,)
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [cp.norm(x - a, 2) <= 1])
        y = np.random.randn(n,)
        y_var = cp.Variable(shape=(n,))
        prob = cp.Problem(cp.Minimize(sigma(y_var)), [y == y_var])
        prob.solve(solver='CLARABEL')
        actual = prob.value
        expected = a @ y + np.linalg.norm(y, ord=2)
        self.assertLessEqual(abs(actual - expected), 1e-6)
        self.assertLessEqual(abs(prob.objective.expr.value - prob.value), 1e-6)

    def test_rectangular_variable(self) -> None:
        np.random.seed(2)
        rows, cols = 4, 2
        a = np.random.randn(rows, cols)
        x = cp.Variable(shape=(rows, cols))
        sigma = cp.suppfunc(x, [x[:, 0] == 0])
        y = cp.Variable(shape=(rows, cols))
        cons = [sigma(y - a) <= 0]
        objective = cp.Minimize(cp.sum_squares(y.flatten(order='F')))
        prob = cp.Problem(objective, cons)
        prob.solve(solver='CLARABEL')
        expect = np.hstack([np.zeros(shape=(rows, 1)), a[:, [1]]])
        actual = y.value
        self.assertLessEqual(np.linalg.norm(actual - expect, ord=2), 1e-6)
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-6)

    def test_psd_dualcone(self) -> None:
        np.random.seed(5)
        n = 3
        X = cp.Variable(shape=(n, n))
        sigma = cp.suppfunc(X, [X >> 0])
        A = np.random.randn(n, n)
        Y = cp.Variable(shape=(n, n))
        objective = cp.Minimize(cp.norm(A.ravel(order='F') + Y.flatten(order='F')))
        cons = [sigma(Y) <= 0]  # Y is negative definite.
        prob = cp.Problem(objective, cons)
        prob.solve(solver='SCS', eps=1e-8)
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-6)
        eigs = np.linalg.eigh(Y.value)[0]
        self.assertLessEqual(np.max(eigs), 1e-6)

    def test_psd_support_across_solvers(self) -> None:
        X = cp.Variable((2, 2))
        sigma = cp.suppfunc(X, [X >> 0, cp.trace(X) <= 1])
        Y = cp.Variable((2, 2))
        A = np.diag([1.0, 2.0])
        epigraph = cp.Variable()
        prob = cp.Problem(cp.Minimize(epigraph), [sigma(Y) <= epigraph, Y == A])

        for solver in [cp.SCS, cp.CLARABEL]:
            prob.solve(solver=solver)
            self.assertAlmostEqual(epigraph.value, 2.0, places=5)

    def test_psd_native_solver_context(self) -> None:
        X = cp.Variable((2, 2))
        sigma = cp.suppfunc(X, [X >> 0, cp.trace(X) <= 1])
        A = np.array([[0.0, 1.0], [1.0, 0.0]])
        expr = sigma(A)

        epigraph, constraints = suppfunc_canon(
            expr, [cp.Constant(A)], _solver_context(SDPA))
        self.assertEqual(sum(isinstance(con, PSD) for con in constraints), 1)
        self.assertFalse(any(isinstance(con, SvecPSD) for con in constraints))

        prob = cp.Problem(cp.Minimize(epigraph), constraints)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 1.0, places=6)

    def test_unscaled_svec_uses_dual_weights(self) -> None:
        X = cp.Variable((2, 2))
        sigma = cp.suppfunc(X, [X >> 0, cp.trace(X) <= 1])
        expr = sigma(np.eye(2))

        _, constraints = suppfunc_canon(
            expr, [cp.Constant(np.eye(2))], _solver_context(COPT))
        svec_con = next(con for con in constraints if isinstance(con, SvecPSD))
        expr._eta.value = np.ones(expr._eta.size)
        np.testing.assert_allclose(svec_con.args[0].value, [1.0, 0.5, 1.0])

    def test_auxiliary_variable_bound_attribute(self) -> None:
        x = cp.Variable(1)
        aux = cp.Variable(1, nonneg=True)
        sigma = cp.suppfunc(x, [x == aux, aux <= 2])
        y = cp.Variable(1)

        for direction, expected in [(1.0, 2.0), (-1.0, 0.0)]:
            prob = cp.Problem(cp.Minimize(sigma(y)), [y == direction])
            prob.solve(solver=cp.CLARABEL)
            self.assertAlmostEqual(prob.value, expected, places=6)

    def test_discrete_set_descriptions_are_rejected(self) -> None:
        x = cp.Variable()
        z = cp.Variable(2, boolean=True)
        support_functions = [
            cp.suppfunc(x, [x == 2 * cp.sum(z), 2 * cp.sum(z) <= 3]),
            cp.suppfunc(x, [cp.FiniteSet(x, [0, 2])]),
        ]

        for sigma in support_functions:
            y = cp.Variable()
            prob = cp.Problem(cp.Minimize(sigma(y)), [y == 1])
            with self.assertRaisesRegex(SolverError, "mixed-integer set"):
                prob.get_problem_data(solver=cp.CLARABEL)

    def test_vectorized_soc_row_order(self) -> None:
        x = cp.Variable(2)
        X = cp.reshape(x, (1, 2), order='F')
        sigma = cp.suppfunc(x, [cp.SOC(np.array([1.0, 2.0]), X)])
        y = cp.Constant([2.0, 1.0])

        epigraph, constraints = suppfunc_canon(
            sigma(y), [y], _solver_context(CLARABEL))
        prob = cp.Problem(cp.Minimize(epigraph), constraints)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 4.0, places=6)

    def test_vectorized_expcone_row_order(self) -> None:
        x = cp.Variable(2)
        sigma = cp.suppfunc(x, [cp.exp(x) <= np.exp([1.0, 2.0])])
        y = cp.Constant([2.0, 1.0])

        epigraph, constraints = suppfunc_canon(
            sigma(y), [y], _solver_context(CLARABEL))
        prob = cp.Problem(cp.Minimize(epigraph), constraints)
        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(prob.value, 4.0, places=6)

    def test_construction_is_lazy(self) -> None:
        x = cp.Variable(1)
        with patch("cvxpy.transforms.suppfunc._coniclift") as lift:
            with patch("cvxpy.problems.problem.Problem.get_problem_data") as get_data:
                cp.suppfunc(x, [x <= 1])
        lift.assert_not_called()
        get_data.assert_not_called()

    def test_constraints_are_snapshotted(self) -> None:
        x = cp.Variable()
        constraints = [x <= 1]
        sigma = cp.suppfunc(x, constraints)
        constraints.append(x <= 0)
        epigraph = cp.Variable()
        prob = cp.Problem(cp.Minimize(epigraph), [sigma(1) <= epigraph])

        prob.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(epigraph.value, 1.0, places=6)

    def test_svec_psd_canonicalization_is_cached(self) -> None:
        X = cp.Variable((2, 2))
        sigma = cp.suppfunc(X, [X >> 0, cp.trace(X) <= 1])
        Y = cp.Variable((2, 2))
        solver_context = _solver_context(SCS)

        with patch("cvxpy.transforms.suppfunc._coniclift", wraps=_coniclift) as lift:
            _, constraints = suppfunc_canon(sigma(Y), [Y], solver_context)
            suppfunc_canon(sigma(Y), [Y], solver_context)

        self.assertEqual(lift.call_count, 1)
        self.assertEqual(sum(isinstance(con, SvecPSD) for con in constraints), 1)

    def test_scs_compatibility_api(self) -> None:
        x = cp.Variable(1)
        constraints = [x <= 1]
        A, b, K = scs_coniclift(x, constraints)
        selectors = scs_cone_selectors(K)
        self.assertEqual(A.shape[0], b.size)
        self.assertEqual(selectors["nonneg"].size, 1)

        sigma = cp.suppfunc(x, constraints)
        old_A, old_b, old_selectors = sigma.conic_repr_of_set()
        self.assertEqual(old_A.shape, A.shape)
        self.assertEqual(old_b.shape, b.shape)
        np.testing.assert_allclose(old_A.toarray(), A.toarray())
        np.testing.assert_allclose(old_b, b)
        self.assertEqual(old_selectors["nonneg"].size, 1)

        vec = cp.Variable(3)
        self.assertEqual(scs_psdvec_to_psdmat(vec, np.arange(3)).shape, (2, 2))

    def test_power_cone_remains_unsupported(self) -> None:
        x = cp.Variable(3)
        sigma = cp.suppfunc(x, [cp.PowCone3D(x[0], x[1], x[2], 0.5)])
        y = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(sigma(y)), [y == np.ones(3)])

        with self.assertRaisesRegex(NotImplementedError, "power cone"):
            prob.get_problem_data(solver=cp.SCS)

    def test_largest_singvalue(self) -> None:
        np.random.seed(3)
        rows, cols = 3, 4
        A = np.random.randn(rows, cols)
        A_sv = np.linalg.svd(A, compute_uv=False)
        X = cp.Variable(shape=(rows, cols))
        sigma = cp.suppfunc(X, [cp.sigma_max(X) <= 1])
        Y = cp.Variable(shape=(rows, cols))
        cons = [Y == A]
        prob = cp.Problem(cp.Minimize(sigma(Y)), cons)
        prob.solve(solver='SCS', eps=1e-8)
        actual = prob.value
        expect = np.sum(A_sv)
        self.assertLessEqual(abs(actual - expect), 1e-6)

    def test_expcone_1(self) -> None:
        x = cp.Variable(shape=(1,))
        tempcons = [cp.exp(x[0]) <= np.exp(1), cp.exp(-x[0]) <= np.exp(1)]
        sigma = cp.suppfunc(x, tempcons)
        y = cp.Variable(shape=(1,))
        obj_expr = y[0]
        cons = [sigma(y) <= 1]
        # ^ That just means -1 <= y[0] <= 1
        prob = cp.Problem(cp.Minimize(obj_expr), cons)
        prob.solve(solver='CLARABEL')
        viol = cons[0].violation()
        self.assertLessEqual(viol, 1e-6)
        self.assertLessEqual(abs(y.value - (-1)), 1e-6)

    def test_expcone_2(self) -> None:
        x = cp.Variable(shape=(3,))
        tempcons = [cp.sum(x) <= 1.0, cp.sum(x) >= 0.1, x >= 0.01,
                    cp.kl_div(x[1], x[0]) + x[1] - x[0] + x[2] <= 0]
        sigma = cp.suppfunc(x, tempcons)
        y = cp.Variable(shape=(3,))
        a = np.array([-3, -2, -1])  # this is negative of objective in mosek_conif.py example
        expr = -sigma(y)
        objective = cp.Maximize(expr)
        cons = [y == a]
        prob = cp.Problem(objective, cons)
        prob.solve(solver='CLARABEL')
        # Check for expected objective value
        epi_actual = prob.value
        direct_actual = expr.value
        expect = 0.235348211
        self.assertLessEqual(abs(epi_actual - expect), 1e-6)
        self.assertLessEqual(abs(direct_actual - expect), 1e-6)

    def test_basic_lmi(self) -> None:
        np.random.seed(4)
        n = 3
        A = np.random.randn(n, n)
        A = A.T @ A
        X = cp.Variable(shape=(n, n))  # will fail if you try PSD=True, or symmetric=Trues
        sigma = cp.suppfunc(X, [0 << X, cp.lambda_max(X) <= 1])
        Y = cp.Variable(shape=(n, n))
        cons = [Y == A]
        expr = sigma(Y)
        prob = cp.Problem(cp.Minimize(expr), cons)  # opt value of support func would be at X=I.
        prob.solve(solver='SCS', eps=1e-8)
        actual1 = prob.value  # computed with epigraph
        actual2 = expr.value  # computed by evaluating support function, as a maximization problem.
        self.assertLessEqual(abs(actual1 - actual2), 1e-6)
        expect = np.trace(A)
        self.assertLessEqual(abs(actual1 - expect), 1e-4)

    def test_invalid_solver(self) -> None:
        n = 3
        x = cp.Variable(shape=(n,))
        sigma = cp.suppfunc(x, [cp.norm(x - np.random.randn(n,), 2) <= 1])
        y_var = cp.Variable(shape=(n,))
        prob = cp.Problem(cp.Minimize(sigma(y_var)), [np.random.randn(n,) == y_var])
        with self.assertRaises(SolverError):
            prob.solve(solver='OSQP')

    def test_invalid_variable(self) -> None:
        x = cp.Variable(shape=(2, 2), symmetric=True)
        with self.assertRaises(ValueError):
            cp.suppfunc(x, [])

    def test_invalid_constraint(self) -> None:
        x = cp.Variable(shape=(3,))
        a = cp.Parameter(shape=(3,))
        cons = [a @ x == 1]
        with self.assertRaises(ValueError):
            cp.suppfunc(x, cons)

    def test_support_function_atom_metadata_and_validation(self) -> None:
        x = cp.Variable(2)
        sigma = cp.suppfunc(x, [cp.norm(x, 2) <= 1])
        y = cp.Variable(2)
        atom = sigma(y)

        self.assertEqual(atom.variables(), [y])
        self.assertEqual(atom.parameters(), [])
        self.assertEqual(atom.constants(), [])
        self.assertEqual(atom.shape_from_args(), tuple())
        self.assertEqual(atom.sign_from_args(), (False, False))
        self.assertFalse(atom.is_nonneg())
        self.assertFalse(atom.is_nonpos())
        self.assertFalse(atom.is_imag())
        self.assertFalse(atom.is_complex())
        self.assertTrue(atom.is_atom_convex())
        self.assertFalse(atom.is_atom_concave())
        self.assertFalse(atom.is_atom_log_log_convex())
        self.assertFalse(atom.is_atom_log_log_concave())
        self.assertTrue(atom.is_atom_quasiconvex())
        self.assertFalse(atom.is_atom_quasiconcave())
        self.assertFalse(atom.is_incr(0))
        self.assertFalse(atom.is_decr(0))
        self.assertTrue(atom.is_convex())
        self.assertFalse(atom.is_concave())
        self.assertTrue(atom.is_quasiconvex())
        self.assertFalse(atom.is_quasiconcave())

        with pytest.raises(ValueError, match="cannot be complex"):
            sigma(cp.Variable(2, complex=True))
        with pytest.raises(ValueError, match="must be affine"):
            sigma(cp.square(y))
        with pytest.raises(NotImplementedError):
            atom < 1
        with pytest.raises(NotImplementedError):
            atom > 1

    def test_support_function_value_and_gradient_paths(self) -> None:
        x = cp.Variable(1)
        sigma = cp.suppfunc(x, [x <= 1, x >= -1])
        y = cp.Variable(1)
        atom = sigma(y)

        y.value = np.array([2.0])
        self.assertAlmostEqual(atom.value, 2.0, places=4)
        grad = atom._grad([np.array([-2.0])])[0]
        np.testing.assert_allclose(y.value, [2.0])
        np.testing.assert_allclose(grad.toarray(), [[-1.0]], atol=1e-4)

        grad = atom._grad([np.array([2.0])])[0]
        np.testing.assert_allclose(y.value, [2.0])
        np.testing.assert_allclose(grad.toarray(), [[1.0]], atol=1e-4)
