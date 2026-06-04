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
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


def _spd(n, seed):
    """A symmetric positive-definite n x n matrix."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    return M @ M.T + n * np.eye(n)


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestQuadFormDifferentFormats:

    def test_quad_form_dense_sparse_sparse(self):
        # Generate a random non-trivial quadratic program.
        m = 15
        n = 10
        p = 5
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = G @ np.random.randn(n, 1)
        A = np.random.randn(p, n)
        b = np.random.randn(p, 1)
        x = cp.Variable((n, 1))

        constraints = [G @ x <= h,
                       A @ x == b]

        # dense problem
        x.value = None
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        dense_val = x.value

        # CSR problem
        x.value = None
        P_csr = sp.csr_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csr) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csr_val = x.value

        # CSC problem
        x.value = None
        P_csc = sp.csc_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csc) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csc_val = x.value

        assert np.allclose(dense_val, csr_val)
        assert np.allclose(dense_val, csc_val)

    def test_quad_form_dense_sparse_sparse_different_x(self):
        # Generate a random non-trivial quadratic program.
        m = 15
        n = 10
        p = 5
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = G @ np.random.randn(n)
        A = np.random.randn(p, n)
        b = np.random.randn(p)
        x = cp.Variable(n)

        constraints = [G @ x <= h,
                       A @ x == b]

        # dense problem
        x.value = None
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        dense_val = x.value

        # CSR problem
        x.value = None
        P_csr = sp.csr_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csr) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csr_val = x.value

        # CSC problem
        x.value = None
        P_csc = sp.csc_matrix(P)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P_csc) + q.T @ x),
                        constraints)
        prob.solve(nlp=True, verbose=False)
        csc_val = x.value

        assert np.allclose(dense_val, csr_val)
        assert np.allclose(dense_val, csc_val)


class TestQuadFormDiffEngine:
    """Derivative-engine checks for quad_form x'Px across P formats and a leaf or
    composed x. Uses DerivativeChecker (objective/gradient/jacobian/hessian), which
    builds the C diff-engine problem directly and needs no NLP solver.
    """

    @staticmethod
    def _var(n, seed):
        x = cp.Variable(n, bounds=[-1, 1])
        x.value = np.random.default_rng(seed).uniform(-0.9, 0.9, n)
        return x

    def test_quad_form_dense_P(self):
        # Dense constant P over a leaf variable (the permuted_dense fast path).
        n = 6
        P = _spd(n, seed=0)
        x = self._var(n, seed=10)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)))
        DerivativeChecker(prob).run_and_assert()

    def test_quad_form_sparse_P(self):
        # Sparse constant P over a leaf variable (the CSR path).
        n = 6
        P = sp.csr_matrix(_spd(n, seed=1))
        x = self._var(n, seed=11)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)))
        DerivativeChecker(prob).run_and_assert()

    def test_quad_form_parametric_P(self):
        # Parametric P (re-evaluated each solve) over a leaf variable.
        n = 6
        P = cp.Parameter((n, n), PSD=True, value=_spd(n, seed=2))
        x = self._var(n, seed=12)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P)))
        DerivativeChecker(prob).run_and_assert()

    def test_quad_form_dense_P_composition(self):
        # Dense P over a composed (sliced) argument x[:k], not a single variable.
        n, k = 6, 3
        P = _spd(k, seed=3)
        x = self._var(n, seed=13)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x[:k], P)))
        DerivativeChecker(prob).run_and_assert()

    def test_quad_form_sparse_P_composition(self):
        # Sparse P over a composed (sliced) argument x[:k].
        n, k = 6, 3
        P = sp.csr_matrix(_spd(k, seed=4))
        x = self._var(n, seed=14)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x[:k], P)))
        DerivativeChecker(prob).run_and_assert()
