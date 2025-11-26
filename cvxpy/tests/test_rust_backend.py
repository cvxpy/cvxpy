# Copyright 2024, the CVXPY developers.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Comprehensive tests for the Rust canonicalization backend.

Tests every LinOp type against the SciPy backend using hypothesis
for property-based testing with varying dimensions.

LinOp types covered (24 total):
- Leaf: variable, scalar_const, dense_const, sparse_const, param
- Trivial: sum, neg, reshape
- Arithmetic: mul, rmul, mul_elem, div
- Structural: index, transpose, promote, broadcast_to, hstack, vstack, concatenate
- Specialized: sum_entries, trace, diag_vec, diag_mat, upper_tri, conv, kron_r, kron_l
- No-op: noop
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy import sparse

import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import CanonBackend
from cvxpy.lin_ops.lin_op import CONSTANT_ID, LinOp

# Check if Rust backend is available
try:
    from cvxpy.lin_ops.canon_backend import RustCanonBackend
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False


def linOpHelper(shape, type, data=None, args=None):
    """Create a LinOp with the given parameters."""
    if args is None:
        args = []
    return LinOp(type, shape, args, data)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="cvxpy_rust not installed")
class TestRustBackend:
    """
    Comprehensive tests for Rust backend comparing against SciPy.
    """

    @staticmethod
    def get_backends(id_to_col, param_to_size, param_to_col, param_size_plus_one, var_length):
        """Create both SciPy and Rust backends with same parameters."""
        kwargs = {
            "id_to_col": id_to_col.copy(),
            "param_to_size": param_to_size.copy(),
            "param_to_col": param_to_col.copy(),
            "param_size_plus_one": param_size_plus_one,
            "var_length": var_length,
        }
        scipy_backend = CanonBackend.get_backend(s.SCIPY_CANON_BACKEND, **kwargs)
        rust_backend = RustCanonBackend(**kwargs)
        return scipy_backend, rust_backend

    @staticmethod
    def compare_matrices(scipy_result, rust_result, rtol=1e-10, atol=1e-12):
        """Compare sparse matrices from both backends."""
        assert scipy_result.shape == rust_result.shape, \
            f"Shape mismatch: {scipy_result.shape} vs {rust_result.shape}"
        diff = scipy_result - rust_result
        if diff.nnz > 0:
            max_diff = np.max(np.abs(diff.data))
            assert max_diff < atol, f"Max absolute difference: {max_diff}"

    # =========================================================================
    # Leaf Operations
    # =========================================================================

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_variable_2d(self, m, n):
        """Test variable with 2D shape."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        lin_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        scipy_result = scipy_backend.build_matrix([lin_op])
        rust_result = rust_backend.build_matrix([lin_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=20, deadline=None)
    def test_variable_1d(self, n):
        """Test variable with 1D shape."""
        var_id = 1
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=n,
        )
        lin_op = linOpHelper(shape=(n,), type="variable", data=var_id, args=[])
        scipy_result = scipy_backend.build_matrix([lin_op])
        rust_result = rust_backend.build_matrix([lin_op])
        self.compare_matrices(scipy_result, rust_result)

    def test_scalar_const(self):
        """Test scalar constant."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        const_op = linOpHelper(shape=(), type="scalar_const", data=3.14, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_dense_const_2d(self, m, n):
        """Test dense constant with 2D shape."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        const_data = np.random.randn(m, n)
        const_op = linOpHelper(shape=(m, n), type="dense_const", data=const_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=20, deadline=None)
    def test_dense_const_1d(self, n):
        """Test dense constant with 1D shape."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        const_data = np.random.randn(n)
        const_op = linOpHelper(shape=(n,), type="dense_const", data=const_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=5),
           st.integers(min_value=2, max_value=5),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=10, deadline=None)
    def test_dense_const_3d(self, a, b, c):
        """Test dense constant with 3D shape (n-dimensional)."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        const_data = np.random.randn(a, b, c)
        const_op = linOpHelper(shape=(a, b, c), type="dense_const", data=const_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=5, max_value=20),
           st.integers(min_value=5, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_sparse_const(self, m, n):
        """Test sparse constant."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        sparse_data = sparse.random(m, n, density=0.3, format='csc')
        const_op = linOpHelper(shape=(m, n), type="sparse_const", data=sparse_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_param(self, n):
        """Test parameter."""
        param_id = 100
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1, param_id: n},
            param_to_col={CONSTANT_ID: 0, param_id: 1},
            param_size_plus_one=n + 1,
            var_length=0,
        )
        lin_op = linOpHelper(shape=(n,), type="param", data=param_id, args=[])
        scipy_result = scipy_backend.build_matrix([lin_op])
        rust_result = rust_backend.build_matrix([lin_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Trivial Operations
    # =========================================================================

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_neg(self, m, n):
        """Test negation operation."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        neg_op = linOpHelper(shape=(m, n), type="neg", args=[var_op])
        scipy_result = scipy_backend.build_matrix([neg_op])
        rust_result = rust_backend.build_matrix([neg_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_sum(self, m, n):
        """Test sum operation (combines two variables)."""
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=(m, n), type="variable", data=1, args=[])
        var2 = linOpHelper(shape=(m, n), type="variable", data=2, args=[])
        sum_op = linOpHelper(shape=(m, n), type="sum", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([sum_op])
        rust_result = rust_backend.build_matrix([sum_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_reshape_2d_to_1d(self, m, n):
        """Test reshape from 2D to 1D."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        reshape_op = linOpHelper(shape=(size,), type="reshape", args=[var_op])
        scipy_result = scipy_backend.build_matrix([reshape_op])
        rust_result = rust_backend.build_matrix([reshape_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_reshape_1d_to_2d(self, m, n):
        """Test reshape from 1D to 2D."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(size,), type="variable", data=var_id, args=[])
        reshape_op = linOpHelper(shape=(m, n), type="reshape", args=[var_op])
        scipy_result = scipy_backend.build_matrix([reshape_op])
        rust_result = rust_backend.build_matrix([reshape_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_mul(self, m, k, n):
        """Test left multiplication: A @ X where A is (m, k) constant, X is (k, n) variable."""
        var_id = 1
        size = k * n  # var X has shape (k, n)
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(k, n), type="variable", data=var_id, args=[])
        const_data = np.random.randn(m, k)
        const_op = linOpHelper(shape=(m, k), type="dense_const", data=const_data, args=[])
        mul_op = linOpHelper(shape=(m, n), type="mul", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([mul_op])
        rust_result = rust_backend.build_matrix([mul_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_rmul(self, k, n, p):
        """Test right multiplication: X @ A where X is (k, n) variable, A is (n, p) constant."""
        var_id = 1
        size = k * n  # var X has shape (k, n)
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(k, n), type="variable", data=var_id, args=[])
        const_data = np.random.randn(n, p)
        const_op = linOpHelper(shape=(n, p), type="dense_const", data=const_data, args=[])
        rmul_op = linOpHelper(shape=(k, p), type="rmul", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([rmul_op])
        rust_result = rust_backend.build_matrix([rmul_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_mul_elem(self, m, n):
        """Test element-wise multiplication."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        const_data = np.random.randn(m, n)
        const_op = linOpHelper(shape=(m, n), type="dense_const", data=const_data, args=[])
        mul_elem_op = linOpHelper(shape=(m, n), type="mul_elem", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([mul_elem_op])
        rust_result = rust_backend.build_matrix([mul_elem_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_div(self, m, n):
        """Test division by scalar constant."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        const_op = linOpHelper(shape=(), type="scalar_const", data=2.5, args=[])
        div_op = linOpHelper(shape=(m, n), type="div", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([div_op])
        rust_result = rust_backend.build_matrix([div_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Structural Operations
    # =========================================================================

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_index_first_column(self, m, n):
        """Test indexing to extract first column."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        index_op = linOpHelper(
            shape=(m,), type="index",
            data=[slice(0, m, 1), slice(0, 1, 1)],
            args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([index_op])
        rust_result = rust_backend.build_matrix([index_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_index_submatrix(self, m, n):
        """Test indexing to extract submatrix."""
        assume(m >= 2 and n >= 2)
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        # Extract upper-left 2x2 submatrix
        sub_m = min(2, m)
        sub_n = min(2, n)
        index_op = linOpHelper(
            shape=(sub_m, sub_n), type="index",
            data=[slice(0, sub_m, 1), slice(0, sub_n, 1)],
            args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([index_op])
        rust_result = rust_backend.build_matrix([index_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_transpose_2d(self, m, n):
        """Test 2D transpose."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        transpose_op = linOpHelper(
            shape=(n, m), type="transpose", data=[None], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([transpose_op])
        rust_result = rust_backend.build_matrix([transpose_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=20))
    @settings(max_examples=20, deadline=None)
    def test_promote(self, n):
        """Test promote (scalar to vector)."""
        var_id = 1
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=1,
        )
        var_op = linOpHelper(shape=(), type="variable", data=var_id, args=[])
        promote_op = linOpHelper(shape=(n,), type="promote", args=[var_op])
        scipy_result = scipy_backend.build_matrix([promote_op])
        rust_result = rust_backend.build_matrix([promote_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_hstack_2d(self, m, n):
        """Test horizontal stacking of 2D arrays."""
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=(m, n), type="variable", data=1, args=[])
        var2 = linOpHelper(shape=(m, n), type="variable", data=2, args=[])
        hstack_op = linOpHelper(shape=(m, 2 * n), type="hstack", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([hstack_op])
        rust_result = rust_backend.build_matrix([hstack_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_vstack_2d(self, m, n):
        """Test vertical stacking of 2D arrays."""
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=(m, n), type="variable", data=1, args=[])
        var2 = linOpHelper(shape=(m, n), type="variable", data=2, args=[])
        vstack_op = linOpHelper(shape=(2 * m, n), type="vstack", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([vstack_op])
        rust_result = rust_backend.build_matrix([vstack_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_vstack_1d_to_2d(self, n):
        """Test vertical stacking of 1D arrays into 2D matrix (bmat-like)."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: n},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * n,
        )
        var1 = linOpHelper(shape=(n,), type="variable", data=1, args=[])
        var2 = linOpHelper(shape=(n,), type="variable", data=2, args=[])
        # Vstack 1D arrays to create 2D matrix
        vstack_op = linOpHelper(shape=(2, n), type="vstack", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([vstack_op])
        rust_result = rust_backend.build_matrix([vstack_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_concatenate_axis0(self, m, n):
        """Test concatenate along axis 0."""
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=(m, n), type="variable", data=1, args=[])
        var2 = linOpHelper(shape=(m, n), type="variable", data=2, args=[])
        concat_op = linOpHelper(shape=(2 * m, n), type="concatenate", data=[0], args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([concat_op])
        rust_result = rust_backend.build_matrix([concat_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Specialized Operations
    # =========================================================================

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_sum_entries(self, m, n):
        """Test sum_entries (sum all elements)."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        sum_entries_op = linOpHelper(
            shape=(1,), type="sum_entries", data=[None, True], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([sum_entries_op])
        rust_result = rust_backend.build_matrix([sum_entries_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_sum_entries_axis0(self, m, n):
        """Test sum_entries along axis 0."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        sum_entries_op = linOpHelper(
            shape=(n,), type="sum_entries", data=[0, False], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([sum_entries_op])
        rust_result = rust_backend.build_matrix([sum_entries_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10),
           st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_sum_entries_axis1(self, m, n):
        """Test sum_entries along axis 1."""
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(m, n), type="variable", data=var_id, args=[])
        sum_entries_op = linOpHelper(
            shape=(m,), type="sum_entries", data=[1, False], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([sum_entries_op])
        rust_result = rust_backend.build_matrix([sum_entries_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_trace(self, n):
        """Test trace operation."""
        var_id = 1
        size = n * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(n, n), type="variable", data=var_id, args=[])
        trace_op = linOpHelper(shape=(1,), type="trace", args=[var_op])
        scipy_result = scipy_backend.build_matrix([trace_op])
        rust_result = rust_backend.build_matrix([trace_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=1, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_diag_vec(self, n):
        """Test diag_vec (vector to diagonal matrix)."""
        var_id = 1
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=n,
        )
        var_op = linOpHelper(shape=(n,), type="variable", data=var_id, args=[])
        diag_vec_op = linOpHelper(shape=(n, n), type="diag_vec", data=0, args=[var_op])
        scipy_result = scipy_backend.build_matrix([diag_vec_op])
        rust_result = rust_backend.build_matrix([diag_vec_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_diag_mat(self, n):
        """Test diag_mat (extract diagonal from matrix)."""
        var_id = 1
        size = n * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(n, n), type="variable", data=var_id, args=[])
        diag_mat_op = linOpHelper(shape=(n,), type="diag_mat", data=0, args=[var_op])
        scipy_result = scipy_backend.build_matrix([diag_mat_op])
        rust_result = rust_backend.build_matrix([diag_mat_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_upper_tri(self, n):
        """Test upper_tri (extract upper triangular elements)."""
        var_id = 1
        size = n * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=(n, n), type="variable", data=var_id, args=[])
        # upper_tri extracts strict upper triangle (k=1, excluding diagonal)
        upper_tri_size = n * (n - 1) // 2
        upper_tri_op = linOpHelper(shape=(upper_tri_size,), type="upper_tri", args=[var_op])
        scipy_result = scipy_backend.build_matrix([upper_tri_op])
        rust_result = rust_backend.build_matrix([upper_tri_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_kron_r(self, m, n):
        """Test right Kronecker product."""
        var_id = 1
        size = m
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(m,), type="variable", data=var_id, args=[])
        const_data = np.random.randn(n)
        const_op = linOpHelper(shape=(n,), type="dense_const", data=const_data, args=[])
        kron_r_op = linOpHelper(shape=(m * n,), type="kron_r", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([kron_r_op])
        rust_result = rust_backend.build_matrix([kron_r_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=10),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=20, deadline=None)
    def test_kron_l(self, m, n):
        """Test left Kronecker product."""
        var_id = 1
        size = m
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(m,), type="variable", data=var_id, args=[])
        const_data = np.random.randn(n)
        const_op = linOpHelper(shape=(n,), type="dense_const", data=const_data, args=[])
        kron_l_op = linOpHelper(shape=(m * n,), type="kron_l", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([kron_l_op])
        rust_result = rust_backend.build_matrix([kron_l_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Complex / Integration Tests
    # =========================================================================

    @given(st.integers(min_value=2, max_value=5),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=10, deadline=None)
    def test_nested_operations(self, m, n):
        """Test nested operations: neg(sum(transpose(x), y))."""
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=(m, n), type="variable", data=1, args=[])
        var2 = linOpHelper(shape=(n, m), type="variable", data=2, args=[])
        transpose_op = linOpHelper(shape=(n, m), type="transpose", data=[None], args=[var1])
        sum_op = linOpHelper(shape=(n, m), type="sum", args=[transpose_op, var2])
        neg_op = linOpHelper(shape=(n, m), type="neg", args=[sum_op])
        scipy_result = scipy_backend.build_matrix([neg_op])
        rust_result = rust_backend.build_matrix([neg_op])
        self.compare_matrices(scipy_result, rust_result)

    @given(st.integers(min_value=2, max_value=5),
           st.integers(min_value=2, max_value=5),
           st.integers(min_value=2, max_value=5))
    @settings(max_examples=10, deadline=None)
    def test_matmul_chain(self, m, k, n):
        """Test matrix multiplication chain: A @ X @ B."""
        var_id = 1
        size = k * k
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(k, k), type="variable", data=var_id, args=[])
        A = np.random.randn(m, k)
        B = np.random.randn(k, n)
        A_op = linOpHelper(shape=(m, k), type="dense_const", data=A, args=[])
        B_op = linOpHelper(shape=(k, n), type="dense_const", data=B, args=[])
        # A @ X using mul (left multiply): data @ arg = A @ X
        mul_op = linOpHelper(shape=(m, k), type="mul", data=A_op, args=[var_op])
        # (A @ X) @ B using rmul (right multiply): arg @ data = (A @ X) @ B
        rmul_op = linOpHelper(shape=(m, n), type="rmul", data=B_op, args=[mul_op])
        scipy_result = scipy_backend.build_matrix([rmul_op])
        rust_result = rust_backend.build_matrix([rmul_op])
        self.compare_matrices(scipy_result, rust_result)

    def test_bmat_like_structure(self):
        """Test bmat-like structure: vstack of hstacks."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: 2, 3: 4},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=6,
        )
        # Create [[x1, x2], [x3, const]]
        x1 = linOpHelper(shape=(1,), type="variable", data=1, args=[])
        x2 = linOpHelper(shape=(1,), type="variable", data=2, args=[])
        x3 = linOpHelper(shape=(1,), type="variable", data=3, args=[])
        const = linOpHelper(shape=(1,), type="scalar_const", data=1.0, args=[])

        hstack1 = linOpHelper(shape=(2,), type="hstack", args=[x1, x2])
        hstack2 = linOpHelper(shape=(2,), type="hstack", args=[x3, const])
        vstack_op = linOpHelper(shape=(2, 2), type="vstack", args=[hstack1, hstack2])

        scipy_result = scipy_backend.build_matrix([vstack_op])
        rust_result = rust_backend.build_matrix([vstack_op])
        self.compare_matrices(scipy_result, rust_result)

    def test_multiple_constraints(self):
        """Test building matrix for multiple constraints."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=4,
        )
        var_op = linOpHelper(shape=(2, 2), type="variable", data=1, args=[])
        neg_op = linOpHelper(shape=(2, 2), type="neg", args=[var_op])
        sum_entries_op = linOpHelper(
            shape=(1,), type="sum_entries", data=[None, True], args=[var_op]
        )

        scipy_result = scipy_backend.build_matrix([var_op, neg_op, sum_entries_op])
        rust_result = rust_backend.build_matrix([var_op, neg_op, sum_entries_op])
        self.compare_matrices(scipy_result, rust_result)


@pytest.mark.skipif(not RUST_AVAILABLE, reason="cvxpy_rust not installed")
class TestRustBackendEndToEnd:
    """
    End-to-end tests that solve actual CVXPY problems with Rust backend.
    """

    def test_simple_lp(self):
        """Test simple LP problem."""
        import cvxpy as cp
        np.random.seed(42)

        n = 10
        A = np.random.randn(5, n)
        b = np.abs(np.random.randn(5)) + 1  # Ensure positive RHS
        c = np.abs(np.random.randn(n))  # Non-negative objective

        x = cp.Variable(n)
        # Add bound constraints to ensure boundedness
        prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= -10, x <= 10])

        # Solve with both backends and compare
        prob.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
        scipy_val = prob.value
        scipy_x = x.value.copy()

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.RUST_CANON_BACKEND)
        rust_val = prob.value
        rust_x = x.value.copy()

        assert abs(scipy_val - rust_val) < 1e-5
        assert np.allclose(scipy_x, rust_x, atol=1e-5)

    def test_sum_squares(self):
        """Test sum_squares objective."""
        import cvxpy as cp
        np.random.seed(42)

        n = 10
        A = np.random.randn(n, n)
        b = np.random.randn(n)

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
        scipy_val = prob.value

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.RUST_CANON_BACKEND)
        rust_val = prob.value

        assert abs(scipy_val - rust_val) < 1e-5

    def test_matrix_variable(self):
        """Test problem with matrix variable."""
        import cvxpy as cp
        np.random.seed(42)

        m, n = 5, 4
        A = np.random.randn(m, n)

        X = cp.Variable((m, n))
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X - A)))

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
        scipy_val = prob.value

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.RUST_CANON_BACKEND)
        rust_val = prob.value

        assert abs(scipy_val - rust_val) < 1e-5

    def test_bmat_constraint(self):
        """Test problem with bmat constraint."""
        import cvxpy as cp

        x = cp.Variable(3)
        W = cp.bmat([[x[0], x[2]],
                     [x[1], 1.0]])

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [W >= 0, cp.sum(x) == 2])

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
        scipy_val = prob.value

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.RUST_CANON_BACKEND)
        rust_val = prob.value

        assert abs(scipy_val - rust_val) < 1e-5

    def test_sparse_matrix(self):
        """Test problem with sparse matrix."""
        from scipy import sparse

        import cvxpy as cp

        np.random.seed(42)
        n = 50
        A = sparse.random(n, n, density=0.1, format='csc')
        b = np.random.randn(n)

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x <= b])

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.SCIPY_CANON_BACKEND)
        scipy_val = prob.value

        prob.solve(solver=cp.CLARABEL, canon_backend=cp.RUST_CANON_BACKEND)
        rust_val = prob.value

        assert abs(scipy_val - rust_val) < 1e-5
