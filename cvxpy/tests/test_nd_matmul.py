"""
Comprehensive tests for ND matrix multiplication in CVXPY.

Tests cover:
1. Core functionality (2D constant @ higher-dim variable)
2. Parametric tests (Parameter @ Variable)
3. Broadcasting (batch dimension size 1)
4. Edge cases (B=1, non-square, n=1, k=1, large batch, errors)

All tests are parametrized by backend for cross-backend consistency.

Copyright 2025, the CVXPY authors.

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

import cvxpy as cp

BACKENDS = [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND]


@pytest.fixture(autouse=True)
def seed_rng():
    """Consistent seeding for all tests."""
    np.random.seed(42)


class TestNDMatmul:
    """Core ND matmul functionality: 2D constant @ higher-dim variable."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_2d_const_3d_var(self, backend):
        """Test (m,k) @ (B,k,n) -> (B,m,n)."""
        B, m, k, n = 2, 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        expr = C @ X
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_2d_const_4d_var(self, backend):
        """Test (m,k) @ (B1,B2,k,n) -> (B1,B2,m,n)."""
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        C = np.random.randn(m, k)
        X = cp.Variable((B1, B2, k, n))

        expr = C @ X
        assert expr.shape == (B1, B2, m, n)

        target = np.random.randn(B1, B2, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        # Verify result
        result = np.zeros((B1, B2, m, n))
        for i in range(B1):
            for j in range(B2):
                result[i, j] = C @ X.value[i, j]
        error = np.linalg.norm(result - target) / np.linalg.norm(target)
        assert error < 1e-4

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_2d_const_5d_var(self, backend):
        """Test (m,k) @ (B1,B2,B3,k,n) -> (B1,B2,B3,m,n)."""
        B1, B2, B3, m, k, n = 2, 2, 2, 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((B1, B2, B3, k, n))

        expr = C @ X
        assert expr.shape == (B1, B2, B3, m, n)

        target = np.random.randn(B1, B2, B3, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_varying_const(self, backend):
        """Test (B,m,k) @ (B,k,n) -> (B,m,n) with batch-varying constant."""
        B, m, k, n = 3, 4, 5, 6
        C = np.random.randn(B, m, k)
        X = cp.Variable((B, k, n))

        expr = C @ X
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL


class TestNDMatmulParametric:
    """Test ND matmul with Parameters."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parametric_2d_param_3d_var(self, backend):
        """Test P (m,k) @ X (B,k,n) where P is a Parameter."""
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)

        X = cp.Variable((B, k, n))
        expr = P @ X

        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        expected = P.value @ X.value
        np.testing.assert_allclose(expr.value, expected, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parametric_reoptimization(self, backend):
        """Test that changing parameter value gives correct result."""
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        X = cp.Variable((B, k, n))

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target)))

        # First solve
        P.value = np.eye(m, k)
        prob.solve(canon_backend=backend)
        result1 = X.value.copy()

        # Second solve with different parameter
        P.value = 2 * np.eye(m, k)
        prob.solve(canon_backend=backend)
        result2 = X.value.copy()

        # Results should differ
        assert not np.allclose(result1, result2)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parametric_achieves_target(self, backend):
        """Verify parametric ND matmul achieves an achievable target."""
        B, m, k, n = 2, 3, 4, 5

        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        # Create achievable target
        X_true = np.random.randn(B, k, n)
        target = P.value @ X_true

        prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        # Since target is achievable, optimal value should be near 0
        assert prob.value < 1e-6


class TestNDMatmulBroadcasting:
    """Test batch dimension broadcasting for ND matmul."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_broadcast_const_batch_1(self, backend):
        """Test (1, m, k) @ (B, k, n) broadcasts const to (B, m, n)."""
        B, m, k, n = 3, 4, 5, 6
        C = np.random.randn(1, m, k)
        X = cp.Variable((B, k, n))
        expr = C @ X
        assert expr.shape == (B, m, n)

        X_true = np.random.randn(B, k, n)
        target = C @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(C @ X.value, target, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_broadcast_var_batch_1(self, backend):
        """Test (B, m, k) @ (1, k, n) broadcasts var to (B, m, n)."""
        B, m, k, n = 3, 4, 5, 6
        C = np.random.randn(B, m, k)
        X = cp.Variable((1, k, n))
        expr = C @ X
        assert expr.shape == (B, m, n)

        X_true = np.random.randn(1, k, n)
        target = C @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(C @ X.value, target, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_broadcast_both_batch_dims(self, backend):
        """Test (B1, 1, m, k) @ (1, B2, k, n) broadcasts to (B1, B2, m, n)."""
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        C = np.random.randn(B1, 1, m, k)
        X = cp.Variable((1, B2, k, n))
        expr = C @ X
        assert expr.shape == (B1, B2, m, n)

        X_true = np.random.randn(1, B2, k, n)
        target = C @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(C @ X.value, target, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_broadcast_with_parameter(self, backend):
        """Test broadcast with Parameter @ Variable."""
        B, m, k, n = 3, 4, 5, 6
        P = cp.Parameter((1, m, k))
        P.value = np.random.randn(1, m, k)
        X = cp.Variable((B, k, n))
        expr = P @ X
        assert expr.shape == (B, m, n)

        X_true = np.random.randn(B, k, n)
        target = P.value @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(P.value @ X.value, target, rtol=1e-5)


class TestNDMatmulEdgeCases:
    """Test edge cases for ND matmul."""

    def test_incompatible_batch_dimensions(self):
        """Test that incompatible batch dimensions raise ValueError."""
        C = np.random.randn(3, 4, 5)  # batch=3
        X = cp.Variable((2, 5, 6))    # batch=2

        with pytest.raises(ValueError, match="Incompatible dimensions"):
            C @ X

    def test_incompatible_batch_dimensions_higher_dim(self):
        """Test incompatible batch dims with higher-dimensional arrays."""
        C = np.random.randn(2, 3, 4, 5)  # batch=(2,3)
        X = cp.Variable((2, 4, 5, 6))    # batch=(2,4) - incompatible

        with pytest.raises(ValueError, match="Incompatible dimensions"):
            C @ X

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_size_one(self, backend):
        """Test with B=1 batch dimension."""
        m, k, n = 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((1, k, n))

        expr = C @ X
        assert expr.shape == (1, m, n)

        target = np.random.randn(1, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_non_square_matrices(self, backend):
        """Test with non-square constant matrix."""
        B, m, k, n = 2, 7, 3, 11  # Non-square, different sizes
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        expr = C @ X
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_single_element_last_dim(self, backend):
        """Test with n=1 (column vector result)."""
        B, m, k = 2, 3, 4
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, 1))

        expr = C @ X
        assert expr.shape == (B, m, 1)

        target = np.random.randn(B, m, 1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_single_element_inner_dim(self, backend):
        """Test with k=1 (rank-1 matrices)."""
        B, m, n = 2, 3, 4
        C = np.random.randn(m, 1)
        X = cp.Variable((B, 1, n))

        expr = C @ X
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_large_batch_dimension(self, backend):
        """Test with larger batch dimension."""
        B, m, k, n = 10, 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        expr = C @ X
        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_varying_with_batch_one(self, backend):
        """Test batch-varying constant with B=1."""
        m, k, n = 3, 4, 5
        C = np.random.randn(1, m, k)
        X = cp.Variable((1, k, n))

        expr = C @ X
        assert expr.shape == (1, m, n)

        target = np.random.randn(1, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL


class TestNDRmul:
    """Core ND rmul functionality: higher-dim variable @ constant."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_3d_var_2d_const(self, backend):
        """Test (B,m,k) @ (k,n) -> (B,m,n)."""
        B, m, k, n = 2, 3, 4, 5
        X = cp.Variable((B, m, k))
        C = np.random.randn(k, n)

        expr = X @ C
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        # Verify result
        for b in range(B):
            expected = X.value[b] @ C
            np.testing.assert_allclose(expr.value[b], expected, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_4d_var_2d_const(self, backend):
        """Test (B1,B2,m,k) @ (k,n) -> (B1,B2,m,n)."""
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        X = cp.Variable((B1, B2, m, k))
        C = np.random.randn(k, n)

        expr = X @ C
        assert expr.shape == (B1, B2, m, n)

        # Use an achievable target (X_true @ C)
        X_true = np.random.randn(B1, B2, m, k)
        target = X_true @ C
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        # Verify the solution matches X_true
        np.testing.assert_allclose(X.value, X_true, rtol=1e-4)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_varying_const(self, backend):
        """Test (B,m,k) @ (B,k,n) -> (B,m,n) with batch-varying constant."""
        B, m, k, n = 3, 4, 5, 6
        X = cp.Variable((B, m, k))
        C = np.random.randn(B, k, n)

        expr = X @ C
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        # Verify result
        for b in range(B):
            expected = X.value[b] @ C[b]
            np.testing.assert_allclose(expr.value[b], expected, rtol=1e-5)


class TestNDRmulParametric:
    """Test ND rmul with Parameters."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parametric_3d_var_2d_param(self, backend):
        """Test X (B,m,k) @ P (k,n) where P is a Parameter."""
        B, m, k, n = 2, 3, 4, 5
        X = cp.Variable((B, m, k))
        P = cp.Parameter((k, n))
        P.value = np.random.randn(k, n)

        expr = X @ P
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        expected = X.value @ P.value
        np.testing.assert_allclose(expr.value, expected, rtol=1e-5)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parametric_reoptimization(self, backend):
        """Test that changing parameter value gives correct result."""
        B, m, k, n = 2, 3, 4, 5
        X = cp.Variable((B, m, k))
        P = cp.Parameter((k, n))

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X @ P - target)))

        P.value = np.eye(k, n)
        prob.solve(canon_backend=backend)
        result1 = X.value.copy()

        P.value = 2 * np.eye(k, n)
        prob.solve(canon_backend=backend)
        result2 = X.value.copy()

        # Results should be different when parameter changes
        assert not np.allclose(result1, result2)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_parametric_achieves_target(self, backend):
        """Verify parametric ND rmul achieves an achievable target."""
        B, m, k, n = 2, 3, 4, 5

        X = cp.Variable((B, m, k))
        P = cp.Parameter((k, n))
        P.value = np.random.randn(k, n)

        # Create achievable target
        X_true = np.random.randn(B, m, k)
        target = X_true @ P.value

        prob = cp.Problem(cp.Minimize(cp.sum_squares(X @ P - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL
        # Since target is achievable, optimal value should be near 0
        assert prob.value < 1e-6


class TestNDRmulBroadcasting:
    """Test batch dimension broadcasting for ND rmul."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_broadcast_const_batch_1(self, backend):
        """Test (B, m, k) @ (1, k, n) broadcasts const to (B, m, n)."""
        B, m, k, n = 3, 4, 5, 6
        X = cp.Variable((B, m, k))
        C = np.random.randn(1, k, n)
        expr = X @ C
        assert expr.shape == (B, m, n)

        X_true = np.random.randn(B, m, k)
        target = X_true @ C
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_broadcast_both_batch_dims(self, backend):
        """Test (B1, 1, m, k) @ (1, B2, k, n) broadcasts to (B1, B2, m, n)."""
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        X = cp.Variable((B1, 1, m, k))
        C = np.random.randn(1, B2, k, n)
        expr = X @ C
        assert expr.shape == (B1, B2, m, n)

        X_true = np.random.randn(B1, 1, m, k)
        target = X_true @ C
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        np.testing.assert_allclose(X.value @ C, target, rtol=1e-5)


class TestNDRmulEdgeCases:
    """Test edge cases for ND rmul."""

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_batch_size_one(self, backend):
        """Test with B=1 batch dimension."""
        m, k, n = 3, 4, 5
        X = cp.Variable((1, m, k))
        C = np.random.randn(k, n)

        expr = X @ C
        assert expr.shape == (1, m, n)

        target = np.random.randn(1, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_n_equals_1(self, backend):
        """Test with n=1 (column vector result)."""
        B, m, k = 2, 3, 4
        X = cp.Variable((B, m, k))
        C = np.random.randn(k, 1)

        expr = X @ C
        assert expr.shape == (B, m, 1)

        target = np.random.randn(B, m, 1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_m_equals_1(self, backend):
        """Test with m=1 (row vector variable)."""
        B, k, n = 2, 4, 5
        X = cp.Variable((B, 1, k))
        C = np.random.randn(k, n)

        expr = X @ C
        assert expr.shape == (B, 1, n)

        target = np.random.randn(B, 1, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_k_equals_1(self, backend):
        """Test with k=1 (rank-1 matrices)."""
        B, m, n = 2, 3, 4
        X = cp.Variable((B, m, 1))
        C = np.random.randn(1, n)

        expr = X @ C
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_large_batch_dimension(self, backend):
        """Test with larger batch dimension."""
        B, m, k, n = 10, 3, 4, 5
        X = cp.Variable((B, m, k))
        C = np.random.randn(k, n)

        expr = X @ C
        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_1d_var_1d_const(self, backend):
        """Test 1D variable @ 1D constant (row vector @ column vector = scalar).

        This is a regression test for the case where both variable and constant
        are 1D vectors, which produces a scalar result. The variable is treated
        as a row vector (1, k) and the constant as a column vector (k, 1).
        """
        k = 5
        x = cp.Variable(k)
        c = np.random.randn(k)

        expr = x @ c
        assert expr.shape == ()  # scalar result

        # Minimize quadratic: (x @ c - target)^2 + ||x||^2
        target = 3.0
        prob = cp.Problem(cp.Minimize((expr - target) ** 2 + cp.sum_squares(x)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL

        # Verify: x @ c should be close to target (within regularization)
        actual = x.value @ c
        assert abs(actual - target) < 1.0  # some tolerance due to regularization
