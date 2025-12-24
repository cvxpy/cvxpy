"""
Comprehensive tests for ND matrix multiplication in CVXPY.

Tests cover:
1. Parametric tests (Parameter @ Variable)
2. Broadcasting tests (2D constant @ higher-dim variable)
3. Edge cases (B=1, non-square, n=1, large batch)
4. Cross-backend consistency

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


class TestNDMatmulParametric:
    """Test ND matmul with Parameters."""

    def test_parametric_2d_const_3d_var(self):
        """Test P (m,k) @ X (B,k,n) where P is a Parameter."""
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)

        X = cp.Variable((B, k, n))
        expr = P @ X

        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()

        assert prob.status == cp.OPTIMAL
        # Verify with current parameter value
        expected = P.value @ X.value
        np.testing.assert_allclose(expr.value, expected, rtol=1e-5)

    def test_parametric_reoptimization(self):
        """Test that changing parameter value gives correct result."""
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        X = cp.Variable((B, k, n))

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target)))

        # First solve
        P.value = np.eye(m, k)
        prob.solve()
        result1 = X.value.copy()

        # Second solve with different parameter
        P.value = 2 * np.eye(m, k)
        prob.solve()
        result2 = X.value.copy()

        # Results should differ
        assert not np.allclose(result1, result2)

    @pytest.mark.parametrize("backend", [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND])
    def test_parametric_both_backends(self, backend):
        """Test parametric ND matmul with both backends."""
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL


class TestNDMatmulBroadcasting:
    """Test broadcasting behavior for ND matmul."""

    def test_2d_const_4d_var(self):
        """Test (m,k) @ (B1,B2,k,n) -> (B1,B2,m,n)."""
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        C = np.random.randn(m, k)
        X = cp.Variable((B1, B2, k, n))

        expr = C @ X
        assert expr.shape == (B1, B2, m, n)

        target = np.random.randn(B1, B2, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()

        assert prob.status == cp.OPTIMAL
        # Verify result
        result = np.zeros((B1, B2, m, n))
        for i in range(B1):
            for j in range(B2):
                result[i, j] = C @ X.value[i, j]
        error = np.linalg.norm(result - target) / np.linalg.norm(target)
        assert error < 1e-4

    def test_2d_const_5d_var(self):
        """Test (m,k) @ (B1,B2,B3,k,n) -> (B1,B2,B3,m,n)."""
        B1, B2, B3, m, k, n = 2, 2, 2, 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((B1, B2, B3, k, n))

        expr = C @ X
        assert expr.shape == (B1, B2, B3, m, n)

        target = np.random.randn(B1, B2, B3, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()

        assert prob.status == cp.OPTIMAL


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

    def test_batch_size_one(self):
        """Test with B=1 batch dimension."""
        m, k, n = 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((1, k, n))

        expr = C @ X
        assert expr.shape == (1, m, n)

        target = np.random.randn(1, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()
        assert prob.status == cp.OPTIMAL

    def test_non_square_matrices(self):
        """Test with non-square constant matrix."""
        B, m, k, n = 2, 7, 3, 11  # Non-square, different sizes
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        expr = C @ X
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()
        assert prob.status == cp.OPTIMAL

    def test_single_element_last_dim(self):
        """Test with n=1 (column vector result)."""
        B, m, k = 2, 3, 4
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, 1))

        expr = C @ X
        assert expr.shape == (B, m, 1)

        target = np.random.randn(B, m, 1)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()
        assert prob.status == cp.OPTIMAL

    def test_large_batch_dimension(self):
        """Test with larger batch dimension."""
        B, m, k, n = 10, 3, 4, 5
        C = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        expr = C @ X
        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()
        assert prob.status == cp.OPTIMAL

    def test_single_element_inner_dim(self):
        """Test with k=1 (rank-1 matrices)."""
        B, m, n = 2, 3, 4
        C = np.random.randn(m, 1)
        X = cp.Variable((B, 1, n))

        expr = C @ X
        assert expr.shape == (B, m, n)

        target = np.random.randn(B, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()
        assert prob.status == cp.OPTIMAL

    def test_batch_varying_with_batch_one(self):
        """Test batch-varying constant with B=1."""
        m, k, n = 3, 4, 5
        C = np.random.randn(1, m, k)
        X = cp.Variable((1, k, n))

        expr = C @ X
        assert expr.shape == (1, m, n)

        target = np.random.randn(1, m, n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve()
        assert prob.status == cp.OPTIMAL


class TestNDMatmulBatchBroadcasting:
    """Test batch dimension broadcasting for ND matmul."""

    @pytest.mark.parametrize("backend", [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND])
    def test_broadcast_const_batch_1(self, backend):
        """Test (1, m, k) @ (B, k, n) broadcasts const to (B, m, k)."""
        np.random.seed(42)
        B, m, k, n = 3, 4, 5, 6
        C = np.random.randn(1, m, k)
        X = cp.Variable((B, k, n))
        expr = C @ X
        assert expr.shape == (B, m, n)

        # Use achievable target
        X_true = np.random.randn(B, k, n)
        target = C @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        # Check output matches (problem may be underdetermined)
        np.testing.assert_allclose(C @ X.value, target, rtol=1e-5)

    @pytest.mark.parametrize("backend", [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND])
    def test_broadcast_var_batch_1(self, backend):
        """Test (B, m, k) @ (1, k, n) broadcasts var to (B, k, n)."""
        np.random.seed(42)
        B, m, k, n = 3, 4, 5, 6
        C = np.random.randn(B, m, k)
        X = cp.Variable((1, k, n))
        expr = C @ X
        assert expr.shape == (B, m, n)

        # Use achievable target
        X_true = np.random.randn(1, k, n)
        target = C @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        # Check output matches (problem may be underdetermined)
        np.testing.assert_allclose(C @ X.value, target, rtol=1e-5)

    @pytest.mark.parametrize("backend", [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND])
    def test_broadcast_both_batch_dims(self, backend):
        """Test (B1, 1, m, k) @ (1, B2, k, n) broadcasts both to (B1, B2, m, k)."""
        np.random.seed(42)
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        C = np.random.randn(B1, 1, m, k)
        X = cp.Variable((1, B2, k, n))
        expr = C @ X
        assert expr.shape == (B1, B2, m, n)

        # Use achievable target
        X_true = np.random.randn(1, B2, k, n)
        target = C @ X_true
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr - target)))
        prob.solve(canon_backend=backend)
        assert prob.status == cp.OPTIMAL
        # Check output matches (problem may be underdetermined)
        np.testing.assert_allclose(C @ X.value, target, rtol=1e-5)

    @pytest.mark.parametrize("backend", [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND])
    def test_broadcast_with_parameter(self, backend):
        """Test broadcast with Parameter @ Variable."""
        np.random.seed(42)
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
        # Check output matches (problem may be underdetermined)
        np.testing.assert_allclose(P.value @ X.value, target, rtol=1e-5)


class TestNDMatmulCrossBackendConsistency:
    """Verify both backends produce identical results."""

    def test_consistent_results_2d_const(self):
        """Both backends should give same result for 2D const @ 3D var."""
        np.random.seed(42)
        B, m, k, n = 2, 3, 4, 5
        C = np.random.randn(m, k)
        target = np.random.randn(B, m, n)

        results = {}
        for backend in [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND]:
            X = cp.Variable((B, k, n))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(C @ X - target)))
            prob.solve(canon_backend=backend)
            results[backend] = X.value.copy()

        np.testing.assert_allclose(
            results[cp.SCIPY_CANON_BACKEND],
            results[cp.COO_CANON_BACKEND],
            rtol=1e-5
        )

    def test_consistent_results_batch_varying(self):
        """Both backends should give same result for batch-varying const."""
        np.random.seed(42)
        B, m, k, n = 3, 4, 5, 6
        C = np.random.randn(B, m, k)
        target = np.random.randn(B, m, n)

        results = {}
        for backend in [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND]:
            X = cp.Variable((B, k, n))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(C @ X - target)))
            prob.solve(canon_backend=backend)
            results[backend] = X.value.copy()

        np.testing.assert_allclose(
            results[cp.SCIPY_CANON_BACKEND],
            results[cp.COO_CANON_BACKEND],
            rtol=1e-5
        )

    def test_consistent_results_4d_var(self):
        """Both backends should give same result for 4D variable."""
        np.random.seed(42)
        B1, B2, m, k, n = 2, 3, 4, 5, 6
        C = np.random.randn(m, k)
        target = np.random.randn(B1, B2, m, n)

        results = {}
        for backend in [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND]:
            X = cp.Variable((B1, B2, k, n))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(C @ X - target)))
            prob.solve(canon_backend=backend)
            results[backend] = X.value.copy()

        np.testing.assert_allclose(
            results[cp.SCIPY_CANON_BACKEND],
            results[cp.COO_CANON_BACKEND],
            rtol=1e-5
        )

    def test_consistent_results_parametric(self):
        """Both backends should give same result for parametric matmul."""
        np.random.seed(42)
        B, m, k, n = 2, 3, 4, 5
        P_val = np.random.randn(m, k)
        target = np.random.randn(B, m, n)

        results = {}
        for backend in [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND]:
            P = cp.Parameter((m, k))
            P.value = P_val
            X = cp.Variable((B, k, n))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target)))
            prob.solve(canon_backend=backend)
            results[backend] = X.value.copy()

        np.testing.assert_allclose(
            results[cp.SCIPY_CANON_BACKEND],
            results[cp.COO_CANON_BACKEND],
            rtol=1e-5
        )


class TestNDMatmulReshapeCorrectness:
    """Tests that verify the reshape operations produce correct matrix structure."""

    @pytest.mark.parametrize("backend", [cp.SCIPY_CANON_BACKEND, cp.COO_CANON_BACKEND])
    def test_parametric_matmul_achieves_target(self, backend):
        """
        Verify that parametric ND matmul actually achieves the target (not just consistent).

        For min ||P @ X - target||^2 with underdetermined system, optimal value should be ~0.
        """
        np.random.seed(123)
        B, m, k, n = 2, 3, 4, 5

        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)
        X = cp.Variable((B, k, n))

        # Create achievable target
        X_true = np.random.randn(B, k, n)
        target = P.value @ X_true

        prob = cp.Problem(cp.Minimize(cp.sum_squares(P @ X - target)))
        prob.solve(canon_backend=backend)

        assert prob.status == cp.OPTIMAL, f"Problem not optimal with {backend}"
        # Since target is achievable, optimal value should be near 0
        assert prob.value < 1e-6, f"Optimal value {prob.value} too large for {backend}"

    def test_get_constant_data_shape_for_broadcast_param(self):
        """
        Test that get_constant_data returns correct matrix structure for broadcast parameter.

        This is the intermediate check that catches reshape bugs.
        """
        from cvxpy.lin_ops.backends.coo_backend import CooCanonBackend

        np.random.seed(42)
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)
        X = cp.Variable((B, k, n))
        expr = P @ X

        obj, _ = expr.canonical_form
        const_linop = obj.data  # broadcast_to

        # Set up backend
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr)))
        variables = prob.variables()
        parameters = prob.parameters()

        var_length = sum(int(np.prod(v.shape)) for v in variables)
        id_to_col = {variables[0].id: 0}
        param_to_size = {p.id: int(np.prod(p.shape)) for p in parameters}
        param_to_col = {p.id: 0 for p in parameters}
        param_size = sum(param_to_size.values())

        backend = CooCanonBackend(
            param_to_size=param_to_size,
            param_to_col=param_to_col,
            param_size_plus_one=param_size + 1,
            var_length=var_length,
            id_to_col=id_to_col
        )

        empty_view = backend.get_empty_view()
        lhs_data, is_param_free = backend.get_constant_data(
            const_linop, 
            empty_view, 
            target_shape=(m, k)
        )

        assert not is_param_free, "Parameter expression should not be param_free"

        # Check that reshaped tensor has correct matrix structure
        for param_id, tensor in lhs_data.items():
            assert tensor.m == m, f"Expected m={m}, got {tensor.m}"
            assert tensor.n == k, f"Expected n={k}, got {tensor.n}"
            assert tensor.nnz == m * k, f"Expected nnz={m*k}, got {tensor.nnz}"
            # Each param_idx should appear exactly once (no broadcast duplication)
            unique_params = np.unique(tensor.param_idx)
            assert len(unique_params) == param_to_size[param_id], \
                f"Expected {param_to_size[param_id]} unique param_idx, got {len(unique_params)}"
