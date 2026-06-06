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
from cvxpy.settings import SPARSE_MATMUL_DENSITY_THRESHOLD
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestMatmulDifferentFormats:

    def test_dense_sparse_sparse(self):
        n = 10
        A = np.random.rand(n, n)
        c = np.random.rand(n, 1)
        x = cp.Variable((n, 1), nonneg=True)
        x0 = np.random.rand(n, 1)
        b = A @ x0

        x.value = 10 * np.ones((n, 1))
        obj = cp.Minimize(c.T @ x)

        # solve problem with dense A
        constraints = [A @ x == b]
        problem = cp.Problem(obj, constraints)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        dense_val = problem.value
        dense_sol = x.value

        x.value = 10 * np.ones((n, 1))

        # solve problem with sparse A CSR
        A_sparse = sp.csr_matrix(A)
        constraints = [A_sparse @ x == b]
        problem = cp.Problem(obj, constraints)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        sparse_val = problem.value
        sparse_sol = x.value

        x.value = 10 * np.ones((n, 1))
        # solve problem with sparse A CSC
        A_sparse = sp.csc_matrix(A)
        constraints = [A_sparse @ x == b]
        problem = cp.Problem(obj, constraints)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        csc_val = problem.value
        csc_sol = x.value

        assert np.allclose(dense_val, sparse_val)
        assert np.allclose(dense_val, csc_val)
        assert np.allclose(dense_sol, sparse_sol)
        assert np.allclose(dense_sol, csc_sol)

    def test_dense_left_matmul(self):
        np.random.seed(0)
        m, n = 4, 4
        A = np.random.rand(m, n)
        X = cp.Variable((n, n), nonneg=True)
        B = np.random.rand(m, n)
        obj = cp.Minimize(cp.sum_squares(A @ X - B))
        constraints = []
        problem = cp.Problem(obj, constraints)
        problem.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_dense_right_matmul(self):
        np.random.seed(0)
        m, n = 4, 4
        A = np.random.rand(m, n)
        X = cp.Variable((n, n), nonneg=True)
        B = np.random.rand(m, n)
        obj = cp.Minimize(cp.sum_squares(X @ A - B))
        constraints = []
        problem = cp.Problem(obj, constraints)
        problem.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_sparse_and_dense_matmul(self):
        np.random.seed(0)
        m, n = 4, 4
        A = np.random.rand(m, n)
        C = sp.random(m, n, density=0.5)
        X = cp.Variable((n, n), nonneg=True)
        B = np.random.rand(m, n)
        obj = cp.Minimize(cp.sum_squares(A @ X @ C - B))
        constraints = []
        problem = cp.Problem(obj, constraints)
        problem.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_sparse_and_dense_matmul2(self):
        np.random.seed(0)
        m, n = 4, 3
        A = np.random.rand(n, m)
        C = sp.random(m, n, density=0.5)
        X = cp.Variable((n, n), nonneg=True)
        B = np.random.rand(m, m)
        obj = cp.Minimize(cp.sum_squares(C @ X @ A - B))
        constraints = []
        problem = cp.Problem(obj, constraints)
        problem.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()


class TestSparseMatmulDispatch:
    # A constant dense left-matmul operand that is mostly zeros is auto-routed to the
    # sparse CSR binding (convert_matmul, SPARSE_MATMUL_DENSITY_THRESHOLD) instead of the
    # dense path, which would build a dense Jacobian/Hessian.
    #
    # These tests observe *which path ran* via the Lagrange Hessian sparsity rather than
    # just numerical correctness (which holds on either path). For sum_squares(A @ x - b)
    # the Hessian is 2 A^T A: the dense path always reports a full lower triangle
    # (n*(n+1)/2 nnz), while the sparse path exposes the true A^T A sparsity. No IPOPT is
    # needed -- only C-problem construction and finite-difference derivative checks.

    @staticmethod
    def _var(n):
        # DerivativeChecker builds an initial point, so the variable needs a value.
        x = cp.Variable(n)
        x.value = np.zeros(n)
        return x

    @staticmethod
    def _hessian_lower_nnz(prob, check=False):
        checker = DerivativeChecker(prob)
        if check:
            assert all(checker.run().values())
        else:
            checker._init_coo()
        return len(checker.hess_rows)

    def test_super_sparse_dense_routes_to_sparse(self):
        # Dense numpy operand well below the density threshold must take the sparse path:
        # its Hessian sparsity matches the explicitly-CSR construction and is far below
        # the full dense triangle. run() also confirms the sparsified path is correct.
        np.random.seed(0)
        n = 40
        full_lower = n * (n + 1) // 2
        density = SPARSE_MATMUL_DENSITY_THRESHOLD / 5  # safely below the threshold
        A_sp = sp.random(n, n, density=density, format='csr', random_state=0)
        A_dense = A_sp.toarray()
        b = np.ones(n)

        prob_dense = cp.Problem(cp.Minimize(cp.sum_squares(A_dense @ self._var(n) - b)))
        dense_nnz = self._hessian_lower_nnz(prob_dense, check=True)

        prob_csr = cp.Problem(cp.Minimize(cp.sum_squares(A_sp @ self._var(n) - b)))
        csr_nnz = self._hessian_lower_nnz(prob_csr)

        assert dense_nnz < full_lower      # not the full dense Hessian
        assert dense_nnz == csr_nnz        # identical to the explicit sparse path

    def test_dense_above_threshold_stays_dense(self):
        # Block-diagonal operand above the density threshold must stay on the dense path.
        # A^T A is genuinely block-sparse, so the explicit-CSR construction yields a
        # smaller Hessian -- which proves the full-triangle assertion below is meaningful.
        np.random.seed(0)
        n, blk = 40, 4
        assert blk / n > SPARSE_MATMUL_DENSITY_THRESHOLD  # density sits above the threshold
        full_lower = n * (n + 1) // 2
        blocks = [np.random.rand(blk, blk) for _ in range(n // blk)]
        A_dense = sp.block_diag(blocks).toarray()  # density = blk / n

        b = np.ones(n)
        prob_dense = cp.Problem(cp.Minimize(cp.sum_squares(A_dense @ self._var(n) - b)))
        dense_nnz = self._hessian_lower_nnz(prob_dense)

        A_sp = sp.csr_matrix(A_dense)
        prob_csr = cp.Problem(cp.Minimize(cp.sum_squares(A_sp @ self._var(n) - b)))
        csr_nnz = self._hessian_lower_nnz(prob_csr)

        assert dense_nnz == full_lower     # dense path: full dense Hessian
        assert csr_nnz < full_lower        # confirms the matrix is structurally sparse

    def test_parametric_super_sparse_not_sparsified(self):
        # A Parameter operand whose current value is super-sparse must NOT be sparsified:
        # freezing its sparsity pattern would corrupt results when the value changes. The
        # param guard (param_node is None) keeps it on the dense path -> full Hessian.
        np.random.seed(0)
        n = 40
        full_lower = n * (n + 1) // 2
        density = SPARSE_MATMUL_DENSITY_THRESHOLD / 5  # safely below the threshold
        A_dense = sp.random(n, n, density=density, format='csr', random_state=0).toarray()

        P = cp.Parameter((n, n))
        P.value = A_dense
        b = np.ones(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ self._var(n) - b)))
        assert self._hessian_lower_nnz(prob) == full_lower
