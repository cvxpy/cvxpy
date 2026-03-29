"""
Copyright 2023, the CVXPY Authors

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

import numpy as np  # noqa F403
import scipy.sparse as sp

from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities import linalg as lau


class TestSparseCholesky(BaseTest):

    def check_gram(self, Lp, A, places=5):
        G = Lp @ Lp.T
        delta = (G - A).toarray().flatten()
        self.assertItemsAlmostEqual(delta, np.zeros(delta.size), places)

    def check_factor(self, L, places=5):
        diag = L.diagonal()
        self.assertTrue(np.all(diag > 0))
        delta = (L - sp.tril(L)).toarray().flatten()
        self.assertItemsAlmostEqual(delta, np.zeros(delta.size), places)

    def test_diagonal(self):
        np.random.seed(0)
        A = sp.csc_array(np.diag(np.random.rand(4)))
        _, L, p = lau.sparse_cholesky(A, 0.0)
        self.check_factor(L)
        self.check_gram(L[p, :], A)

    def test_tridiagonal(self):
        np.random.seed(0)
        n = 5
        diag = np.random.rand(n) + 0.1
        offdiag = np.min(np.abs(diag)) * np.ones(n - 1) / 2
        A = sp.diags_array([offdiag, diag, offdiag], offsets=[-1, 0, 1])
        _, L, p = lau.sparse_cholesky(A, 0.0)
        self.check_factor(L)
        self.check_gram(L[p, :], A)

    def test_generic(self, use_expression=False):
        np.random.seed(0)
        B = np.random.randn(3, 3)
        A = sp.csc_array(B @ B.T)
        if use_expression:
            from cvxpy.expressions.expression import Expression
            A = Expression.cast_to_const(A)
            assert isinstance(A, Expression)
        _, L, p = lau.sparse_cholesky(A)
        self.check_factor(L)
        if use_expression:
            A = A.value
        self.check_gram(L[p, :], A)

    def test_expression(self):
        self.test_generic(use_expression=True)

    def test_rank_deficient(self):
        # PSD but rank-deficient: L should be n-by-k with k < n
        np.random.seed(0)
        B = np.random.randn(4, 2)
        A = sp.csc_array(B @ B.T)
        _, L, p = lau.sparse_cholesky(A)
        self.assertEqual(L.shape[0], 4)
        self.assertEqual(L.shape[1], 2)
        self.check_gram(L[p, :], A)

    def test_rank_deficient_large(self):
        # Larger rank-deficient PSD matrix
        np.random.seed(42)
        n, k = 10, 7
        B = np.random.randn(n, k)
        A = sp.csc_array(B @ B.T)
        _, L, p = lau.sparse_cholesky(A)
        self.assertEqual(L.shape[0], n)
        self.assertEqual(L.shape[1], k)
        self.check_gram(L[p, :], A)

    def test_nsd_rank_deficient(self):
        # NSD rank-deficient matrix
        np.random.seed(0)
        B = np.random.randn(4, 2)
        A = sp.csc_array(-(B @ B.T))
        sign, L, p = lau.sparse_cholesky(A)
        self.assertEqual(sign, -1.0)
        self.assertEqual(L.shape[1], 2)
        # sign * A = -A = B @ B.T, so L[p,:] @ L[p,:].T == -A
        self.check_gram(L[p, :], -A)

    def test_diagonal_with_zeros(self):
        # PSD diagonal matrix with some zero entries
        A = sp.diags_array([3.0, 0.0, 5.0, 0.0], format='csc')
        _, L, p = lau.sparse_cholesky(A, 0.0)
        self.assertEqual(L.shape[1], 2)
        self.check_gram(L[p, :], A)

    def test_zero_matrix(self):
        # All-zero matrix: rank 0
        A = sp.csc_array((4, 4))
        _, L, p = lau.sparse_cholesky(A, 0.0)
        self.assertEqual(L.shape, (4, 0))
        self.check_gram(L[p, :], A)

    def test_nontrivial_permutation(self):
        # Test with a matrix that produces a non-symmetric permutation
        # to verify the permutation handling is correct
        np.random.seed(123)
        n = 6
        B = sp.random(n, n, density=0.4, format='csr', random_state=123)
        A = (B @ B.T + n * sp.eye(n)).tocsc()
        _, L, p = lau.sparse_cholesky(A)
        self.check_factor(L)
        self.check_gram(L[p, :], A)
        # Verify permutation is non-trivial (not identity or simple reversal)
        p_inv = np.argsort(p)
        self.assertFalse(np.array_equal(p, p_inv),
                         "Test requires non-symmetric permutation")

    def test_indefinite_with_zero_diagonal(self):
        # [[1, 2], [2, 0]] has non-negative diagonal but is indefinite.
        # The zero-diagonal row is not all-zero, so we must reject it.
        A = sp.csc_array(np.array([[1.0, 2.0], [2.0, 0.0]]))
        with self.assertRaises(ValueError):
            lau.sparse_cholesky(A, 0.0)

    def test_nonsingular_indefinite(self):
        np.random.seed(0)
        n = 5
        diag = np.random.rand(n) + 0.1
        diag[n-1] = -1
        offdiag = np.min(np.abs(diag)) * np.ones(n - 1) / 2
        A = sp.diags_array([offdiag, diag, offdiag], offsets=[-1, 0, 1])
        with self.assertRaises(ValueError, msg=lau.SparseCholeskyMessages.INDEFINITE):
            lau.sparse_cholesky(A, 0.0)
