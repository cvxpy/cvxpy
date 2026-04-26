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
