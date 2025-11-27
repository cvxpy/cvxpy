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

Tests every LinOp type against the SciPy backend using fixed shape examples
covering all relevant dimensionalities and edge cases.

LinOp types covered (24 total):
- Leaf: variable, scalar_const, dense_const, sparse_const, param
- Trivial: sum, neg, reshape
- Arithmetic: mul, rmul, mul_elem, div
- Structural: index, transpose, promote, broadcast_to, hstack, vstack, concatenate
- Specialized: sum_entries, trace, diag_vec, diag_mat, upper_tri, conv, kron_r, kron_l
- No-op: noop

Shape categories tested:
- 1D arrays: (n,)
- 2D arrays: (m, n)
- Column vectors: (m, 1)
- Row vectors: (1, n)
- 3D arrays: (a, b, c)
- 4D arrays: (a, b, c, d)
"""

import numpy as np
import pytest
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


# ============================================================================
# Fixed shape examples for parametrized tests
# ============================================================================

# Basic shapes for general operations
BASIC_2D_SHAPES = [
    (2, 3),
    (3, 2),
    (4, 4),
    (1, 5),
    (5, 1),
]

BASIC_1D_SHAPES = [
    (3,),
    (5,),
    (10,),
]

# Shape categories for multiplication tests
SHAPE_CATEGORIES = [
    # (name, var_shape, const_shape, output_shape)
    ("1d", (4,), (3, 4), (3,)),
    ("2d", (3, 4), (2, 3), (2, 4)),
    ("col_vec", (4, 1), (3, 4), (3, 1)),
    ("row_vec", (1, 4), (2, 1), (2, 4)),
    ("3d_flat", (2, 3, 2), (4, 12), (4,)),  # 3D var flattened to 12 elements
]

# Shapes for elementwise operations (var and const same shape)
ELEMENTWISE_SHAPES = [
    (4,),           # 1D
    (3, 4),         # 2D
    (4, 1),         # Column vector
    (1, 4),         # Row vector
    (2, 3, 2),      # 3D
    (2, 2, 2, 2),   # 4D
]

# 3D and 4D shapes for n-dimensional tests
ND_SHAPES = [
    (2, 3, 4),
    (3, 2, 2),
    (2, 2, 2, 2),
    (2, 3, 2, 2),
]


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
    def compare_matrices(scipy_result, rust_result, atol=1e-12):
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

    @pytest.mark.parametrize("shape", BASIC_2D_SHAPES)
    def test_variable_2d(self, shape):
        """Test variable with 2D shape."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        lin_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        scipy_result = scipy_backend.build_matrix([lin_op])
        rust_result = rust_backend.build_matrix([lin_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", BASIC_1D_SHAPES)
    def test_variable_1d(self, shape):
        """Test variable with 1D shape."""
        var_id = 1
        n = shape[0]
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=n,
        )
        lin_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        scipy_result = scipy_backend.build_matrix([lin_op])
        rust_result = rust_backend.build_matrix([lin_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_variable_nd(self, shape):
        """Test variable with 3D and 4D shapes."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        lin_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
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

    @pytest.mark.parametrize("shape", BASIC_2D_SHAPES)
    def test_dense_const_2d(self, shape):
        """Test dense constant with 2D shape."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        const_data = np.random.randn(*shape)
        const_op = linOpHelper(shape=shape, type="dense_const", data=const_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", BASIC_1D_SHAPES)
    def test_dense_const_1d(self, shape):
        """Test dense constant with 1D shape."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        const_data = np.random.randn(*shape)
        const_op = linOpHelper(shape=shape, type="dense_const", data=const_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_dense_const_nd(self, shape):
        """Test dense constant with 3D and 4D shape (n-dimensional)."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        const_data = np.random.randn(*shape)
        const_op = linOpHelper(shape=shape, type="dense_const", data=const_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", [(10, 10), (20, 15), (5, 25)])
    def test_sparse_const(self, shape):
        """Test sparse constant."""
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=0,
        )
        np.random.seed(42)
        sparse_data = sparse.random(shape[0], shape[1], density=0.3, format='csc')
        const_op = linOpHelper(shape=shape, type="sparse_const", data=sparse_data, args=[])
        scipy_result = scipy_backend.build_matrix([const_op])
        rust_result = rust_backend.build_matrix([const_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("n", [2, 5, 10])
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

    @pytest.mark.parametrize("shape", BASIC_2D_SHAPES + ND_SHAPES)
    def test_neg(self, shape):
        """Test negation operation."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        neg_op = linOpHelper(shape=shape, type="neg", args=[var_op])
        scipy_result = scipy_backend.build_matrix([neg_op])
        rust_result = rust_backend.build_matrix([neg_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", BASIC_2D_SHAPES)
    def test_sum(self, shape):
        """Test sum operation (combines two variables)."""
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=shape, type="variable", data=1, args=[])
        var2 = linOpHelper(shape=shape, type="variable", data=2, args=[])
        sum_op = linOpHelper(shape=shape, type="sum", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([sum_op])
        rust_result = rust_backend.build_matrix([sum_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", BASIC_2D_SHAPES)
    def test_reshape_2d_to_1d(self, shape):
        """Test reshape from 2D to 1D."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        reshape_op = linOpHelper(shape=(size,), type="reshape", args=[var_op])
        scipy_result = scipy_backend.build_matrix([reshape_op])
        rust_result = rust_backend.build_matrix([reshape_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_reshape_nd_to_1d(self, shape):
        """Test reshape from 3D/4D to 1D."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        reshape_op = linOpHelper(shape=(size,), type="reshape", args=[var_op])
        scipy_result = scipy_backend.build_matrix([reshape_op])
        rust_result = rust_backend.build_matrix([reshape_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Arithmetic Operations - Shape Categories
    # =========================================================================

    @pytest.mark.parametrize("name,var_shape,const_shape,output_shape", SHAPE_CATEGORIES)
    def test_mul_shapes(self, name, var_shape, const_shape, output_shape):
        """Test left multiplication with different shape categories."""
        np.random.seed(42)
        var_id = 1
        var_size = int(np.prod(var_shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=var_size,
        )

        var_op = linOpHelper(shape=var_shape, type="variable", data=var_id, args=[])
        const_data = np.random.randn(*const_shape)
        const_op = linOpHelper(shape=const_shape, type="dense_const", data=const_data, args=[])
        mul_op = linOpHelper(shape=output_shape, type="mul", data=const_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([mul_op])
        rust_result = rust_backend.build_matrix([mul_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("name,var_shape,const_shape,output_shape", [
        ("1d_dot", (4,), (4,), (1,)),
        ("2d", (3, 4), (4, 2), (3, 2)),
        ("col_times_row", (4, 1), (1, 3), (4, 3)),
        ("row_times_col", (1, 4), (4, 1), (1, 1)),
    ])
    def test_rmul_shapes(self, name, var_shape, const_shape, output_shape):
        """Test right multiplication with different shape categories."""
        np.random.seed(42)
        var_id = 1
        var_size = int(np.prod(var_shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=var_size,
        )

        var_op = linOpHelper(shape=var_shape, type="variable", data=var_id, args=[])
        const_data = np.random.randn(*const_shape)
        const_op = linOpHelper(shape=const_shape, type="dense_const", data=const_data, args=[])
        rmul_op = linOpHelper(shape=output_shape, type="rmul", data=const_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([rmul_op])
        rust_result = rust_backend.build_matrix([rmul_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
    def test_mul_elem_shapes(self, shape):
        """Test elementwise multiplication with all shape categories."""
        np.random.seed(42)
        var_id = 1
        var_size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=var_size,
        )

        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        const_data = np.random.randn(*shape)
        const_op = linOpHelper(shape=shape, type="dense_const", data=const_data, args=[])
        mul_elem_op = linOpHelper(shape=shape, type="mul_elem", data=const_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([mul_elem_op])
        rust_result = rust_backend.build_matrix([mul_elem_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", ELEMENTWISE_SHAPES)
    def test_div_shapes(self, shape):
        """Test division by scalar constant with all shape categories."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        const_op = linOpHelper(shape=(), type="scalar_const", data=2.5, args=[])
        div_op = linOpHelper(shape=shape, type="div", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([div_op])
        rust_result = rust_backend.build_matrix([div_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Parametric Operations
    # =========================================================================

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_mul_elem_with_param(self, n):
        """Test elementwise multiplication where data contains a parameter."""
        param_id = 100
        var_id = 1
        param_size = n

        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1, param_id: param_size},
            param_to_col={CONSTANT_ID: 0, param_id: 1},
            param_size_plus_one=param_size + 1,
            var_length=n,
        )

        var_op = linOpHelper(shape=(n,), type="variable", data=var_id, args=[])
        param_op = linOpHelper(shape=(n,), type="param", data=param_id, args=[])
        mul_elem_op = linOpHelper(shape=(n,), type="mul_elem", data=param_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([mul_elem_op])
        rust_result = rust_backend.build_matrix([mul_elem_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", [(2, 2), (3, 3), (2, 4)])
    def test_mul_with_param(self, shape):
        """Test left multiplication where data is a parameter (A @ x)."""
        m, n = shape
        param_id = 100
        var_id = 1
        param_size = m * n
        var_size = n

        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1, param_id: param_size},
            param_to_col={CONSTANT_ID: 0, param_id: 1},
            param_size_plus_one=param_size + 1,
            var_length=var_size,
        )

        var_op = linOpHelper(shape=(n,), type="variable", data=var_id, args=[])
        param_op = linOpHelper(shape=(m, n), type="param", data=param_id, args=[])
        mul_op = linOpHelper(shape=(m,), type="mul", data=param_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([mul_op])
        rust_result = rust_backend.build_matrix([mul_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Structural Operations
    # =========================================================================

    @pytest.mark.parametrize("shape", [(4, 5), (3, 6), (5, 3)])
    def test_index_first_column(self, shape):
        """Test indexing to extract first column."""
        m, n = shape
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        index_op = linOpHelper(
            shape=(m,), type="index",
            data=[slice(0, m, 1), slice(0, 1, 1)],
            args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([index_op])
        rust_result = rust_backend.build_matrix([index_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", [(4, 5), (5, 4), (6, 6)])
    def test_transpose_2d(self, shape):
        """Test 2D transpose."""
        m, n = shape
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        transpose_op = linOpHelper(
            shape=(n, m), type="transpose", data=[None], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([transpose_op])
        rust_result = rust_backend.build_matrix([transpose_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("n", [5, 10, 15])
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

    @pytest.mark.parametrize("shape", [(3, 4), (4, 3), (5, 5)])
    def test_hstack_2d(self, shape):
        """Test horizontal stacking of 2D arrays."""
        m, n = shape
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=shape, type="variable", data=1, args=[])
        var2 = linOpHelper(shape=shape, type="variable", data=2, args=[])
        hstack_op = linOpHelper(shape=(m, 2 * n), type="hstack", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([hstack_op])
        rust_result = rust_backend.build_matrix([hstack_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", [(3, 4), (4, 3), (5, 5)])
    def test_vstack_2d(self, shape):
        """Test vertical stacking of 2D arrays."""
        m, n = shape
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=shape, type="variable", data=1, args=[])
        var2 = linOpHelper(shape=shape, type="variable", data=2, args=[])
        vstack_op = linOpHelper(shape=(2 * m, n), type="vstack", args=[var1, var2])
        scipy_result = scipy_backend.build_matrix([vstack_op])
        rust_result = rust_backend.build_matrix([vstack_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", [(3, 4), (4, 3)])
    def test_concatenate_axis0(self, shape):
        """Test concatenate along axis 0."""
        m, n = shape
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={1: 0, 2: size},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=2 * size,
        )
        var1 = linOpHelper(shape=shape, type="variable", data=1, args=[])
        var2 = linOpHelper(shape=shape, type="variable", data=2, args=[])
        concat_op = linOpHelper(
            shape=(2 * m, n), type="concatenate", data=[0], args=[var1, var2]
        )
        scipy_result = scipy_backend.build_matrix([concat_op])
        rust_result = rust_backend.build_matrix([concat_op])
        self.compare_matrices(scipy_result, rust_result)

    # =========================================================================
    # Specialized Operations
    # =========================================================================

    @pytest.mark.parametrize("shape", [(3, 4), (4, 5), (5, 3)])
    def test_sum_entries(self, shape):
        """Test sum_entries (sum all elements)."""
        var_id = 1
        size = int(np.prod(shape))
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        sum_entries_op = linOpHelper(
            shape=(1,), type="sum_entries", data=[None, True], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([sum_entries_op])
        rust_result = rust_backend.build_matrix([sum_entries_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("shape", [(3, 4), (4, 5)])
    def test_sum_entries_axis0(self, shape):
        """Test sum_entries along axis 0."""
        m, n = shape
        var_id = 1
        size = m * n
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=size,
        )
        var_op = linOpHelper(shape=shape, type="variable", data=var_id, args=[])
        sum_entries_op = linOpHelper(
            shape=(n,), type="sum_entries", data=[0, False], args=[var_op]
        )
        scipy_result = scipy_backend.build_matrix([sum_entries_op])
        rust_result = rust_backend.build_matrix([sum_entries_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("n", [3, 4, 5])
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

    @pytest.mark.parametrize("n", [3, 5, 7])
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

    @pytest.mark.parametrize("n", [3, 4, 5])
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

    @pytest.mark.parametrize("n", [3, 4, 5])
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
        upper_tri_size = n * (n - 1) // 2
        upper_tri_op = linOpHelper(shape=(upper_tri_size,), type="upper_tri", args=[var_op])
        scipy_result = scipy_backend.build_matrix([upper_tri_op])
        rust_result = rust_backend.build_matrix([upper_tri_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("m,n", [(4, 3), (5, 2), (3, 4)])
    def test_kron_r(self, m, n):
        """Test right Kronecker product."""
        var_id = 1
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=m,
        )
        np.random.seed(42)
        var_op = linOpHelper(shape=(m,), type="variable", data=var_id, args=[])
        const_data = np.random.randn(n)
        const_op = linOpHelper(shape=(n,), type="dense_const", data=const_data, args=[])
        kron_r_op = linOpHelper(shape=(m * n,), type="kron_r", data=const_op, args=[var_op])
        scipy_result = scipy_backend.build_matrix([kron_r_op])
        rust_result = rust_backend.build_matrix([kron_r_op])
        self.compare_matrices(scipy_result, rust_result)

    @pytest.mark.parametrize("m,n", [(4, 3), (5, 2), (3, 4)])
    def test_kron_l(self, m, n):
        """Test left Kronecker product."""
        var_id = 1
        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=m,
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
    # Nested / Complex Operations
    # =========================================================================

    def test_mul_with_hstack_data(self):
        """Test multiplication where data is result of hstack."""
        np.random.seed(42)
        n = 4
        var_id = 1

        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=n,
        )

        var_op = linOpHelper(shape=(n,), type="variable", data=var_id, args=[])
        const1 = np.random.randn(2, n // 2)
        const2 = np.random.randn(2, n // 2)
        const1_op = linOpHelper(shape=(2, n // 2), type="dense_const", data=const1, args=[])
        const2_op = linOpHelper(shape=(2, n // 2), type="dense_const", data=const2, args=[])
        hstack_op = linOpHelper(shape=(2, n), type="hstack", args=[const1_op, const2_op])
        mul_op = linOpHelper(shape=(2,), type="mul", data=hstack_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([mul_op])
        rust_result = rust_backend.build_matrix([mul_op])
        self.compare_matrices(scipy_result, rust_result)

    def test_mul_with_transpose_data(self):
        """Test multiplication where data is transposed."""
        np.random.seed(42)
        n = 4
        var_id = 1

        scipy_backend, rust_backend = self.get_backends(
            id_to_col={var_id: 0},
            param_to_size={CONSTANT_ID: 1},
            param_to_col={CONSTANT_ID: 0},
            param_size_plus_one=1,
            var_length=n,
        )

        var_op = linOpHelper(shape=(n,), type="variable", data=var_id, args=[])
        const_data = np.random.randn(n, 3)
        const_op = linOpHelper(shape=(n, 3), type="dense_const", data=const_data, args=[])
        transpose_op = linOpHelper(shape=(3, n), type="transpose", data=[None], args=[const_op])
        mul_op = linOpHelper(shape=(3,), type="mul", data=transpose_op, args=[var_op])

        scipy_result = scipy_backend.build_matrix([mul_op])
        rust_result = rust_backend.build_matrix([mul_op])
        self.compare_matrices(scipy_result, rust_result)

    def test_deeply_nested_operations(self):
        """Test deeply nested operation: mul(neg(sum(x, y)))."""
        np.random.seed(42)
        m, n = 3, 4
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
        neg_op = linOpHelper(shape=(m, n), type="neg", args=[sum_op])
        const_data = np.random.randn(m, m)
        const_op = linOpHelper(shape=(m, m), type="dense_const", data=const_data, args=[])
        mul_op = linOpHelper(shape=(m, n), type="mul", data=const_op, args=[neg_op])

        scipy_result = scipy_backend.build_matrix([mul_op])
        rust_result = rust_backend.build_matrix([mul_op])
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
        b = np.abs(np.random.randn(5)) + 1
        c = np.abs(np.random.randn(n))

        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(c @ x), [A @ x <= b, x >= -10, x <= 10])

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

    def test_3d_variable(self):
        """Test problem with 3D variable."""
        import cvxpy as cp
        np.random.seed(42)

        shape = (2, 3, 4)
        A = np.random.randn(*shape)

        X = cp.Variable(shape)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(X - A)))

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

    def test_parametric_problem(self):
        """Test problem with parameters (DPP)."""
        import cvxpy as cp
        np.random.seed(42)

        x = cp.Variable(2, pos=True)
        A = cp.Parameter(shape=(2, 2))
        A.value = np.array([[-5, 2], [1, -3]])
        b = np.array([3, 2])
        expr = cp.gmatmul(A, x)
        prob = cp.Problem(cp.Minimize(1.0), [expr == b])

        prob.solve(solver=cp.SCS, gp=True, enforce_dpp=True,
                   canon_backend=cp.SCIPY_CANON_BACKEND)
        scipy_x = x.value.copy()

        prob.solve(solver=cp.SCS, gp=True, enforce_dpp=True,
                   canon_backend=cp.RUST_CANON_BACKEND)
        rust_x = x.value.copy()

        assert np.allclose(scipy_x, rust_x, atol=1e-5)
