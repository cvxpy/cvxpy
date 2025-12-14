"""
Exhaustive test suite for canon backends.

Tests all linops across SCIPY, NUMPY, and LAZY backends to ensure:
1. Each backend produces correct output (expected value tests)
2. All backends produce identical output (cross-backend consistency)
3. Parametrized expressions work correctly (DPP tests)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.lin_ops.backends import CanonBackend, TensorRepresentation


@dataclass
class LinOpHelper:
    """Mock LinOp for testing."""
    shape: tuple[int, ...] | None = None
    type: str | None = None
    data: Any = None
    args: list | None = None


# All linops to test (sum excluded - requires special setup not covered here)
LINOPS = [
    "neg", "sum_entries", "mul", "rmul", "mul_elem", "div",
    "promote", "broadcast_to", "reshape", "transpose", "index",
    "diag_vec", "diag_mat", "trace", "upper_tri",
    "hstack", "vstack", "concatenate", "conv", "kron_l", "kron_r",
]

# All backends to test
BACKENDS = [
    s.SCIPY_CANON_BACKEND,
    s.NUMPY_CANON_BACKEND,
    s.COO_CANON_BACKEND,
]


def make_backend(name: str, **kwargs) -> CanonBackend:
    """Create a backend with standard test configuration."""
    defaults = {
        "id_to_col": {1: 0, 2: 4},
        "param_to_size": {-1: 1, 3: 4},
        "param_to_col": {-1: 0, 3: 1},
        "param_size_plus_one": 2,
        "var_length": 8,
    }
    return CanonBackend.get_backend(name, **{**defaults, **kwargs})


def to_dense(tr: TensorRepresentation, shape: tuple[int, int]) -> np.ndarray:
    """Convert TensorRepresentation to dense numpy array (param_offset=0 slice)."""
    mask = tr.parameter_offset == 0
    return sp.coo_matrix(
        (tr.data[mask], (tr.row[mask], tr.col[mask])), shape=shape
    ).toarray()


class TestLinopExpectedValues:
    """Test each linop produces correct output against expected values."""

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_neg(self, backend_name):
        """neg(x) on eye(4) should produce -eye(4)."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.neg(LinOpHelper(), view)
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        np.testing.assert_array_equal(A, -np.eye(4))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_transpose(self, backend_name):
        """transpose on (2,2) variable reorders rows."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lin_op = LinOpHelper((2, 2), data=[None], args=[var])
        out = backend.transpose(lin_op, view)
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_sum_entries(self, backend_name):
        """sum_entries sums all entries to a single row."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lin_op = LinOpHelper(shape=(2,), data=[None, True], args=[var])
        out = backend.sum_entries(lin_op, view)
        A = to_dense(out.get_tensor_representation(0, 1), (1, 2))
        np.testing.assert_array_equal(A, [[1, 1]])

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_promote(self, backend_name):
        """promote repeats scalar to n rows."""
        backend = make_backend(backend_name)
        var = LinOpHelper((1,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.promote(LinOpHelper((3,)), view)
        A = to_dense(out.get_tensor_representation(0, 1), (3, 1))
        np.testing.assert_array_equal(A, [[1], [1], [1]])

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_broadcast_to_rows(self, backend_name):
        """broadcast (3,) -> (2,3) repeats along rows."""
        backend = make_backend(backend_name)
        var = LinOpHelper((3,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lin_op = LinOpHelper((2, 3), data=(2, 3), args=[var])
        out = backend.broadcast_to(lin_op, view)
        A = to_dense(out.get_tensor_representation(0, 3), (6, 3))
        expected = np.array([
            [1, 0, 0], [1, 0, 0],
            [0, 1, 0], [0, 1, 0],
            [0, 0, 1], [0, 0, 1],
        ])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_reshape(self, backend_name):
        """reshape is a no-op on A matrix."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lin_op = LinOpHelper((4,), args=[var])
        out = backend.reshape(lin_op, view)
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        np.testing.assert_array_equal(A, np.eye(4))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_index(self, backend_name):
        """index selects subset of rows."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lin_op = LinOpHelper(data=[slice(0, 2, 1), slice(0, 1, 1)], args=[var])
        out = backend.index(lin_op, view)
        A = to_dense(out.get_tensor_representation(0, 2), (2, 4))
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_diag_vec(self, backend_name):
        """diag_vec embeds vector as diagonal of matrix."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.diag_vec(LinOpHelper(shape=(2, 2), data=0), view)
        A = to_dense(out.get_tensor_representation(0, 4), (4, 2))
        expected = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_diag_mat(self, backend_name):
        """diag_mat extracts diagonal from matrix."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.diag_mat(LinOpHelper(shape=(2, 2), data=0), view)
        A = to_dense(out.get_tensor_representation(0, 2), (2, 4))
        expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_upper_tri(self, backend_name):
        """upper_tri extracts strict upper triangle."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.upper_tri(LinOpHelper(args=[LinOpHelper((2, 2))]), view)
        A = to_dense(out.get_tensor_representation(0, 1), (1, 4))
        expected = np.array([[0, 0, 1, 0]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_trace(self, backend_name):
        """trace sums diagonal entries."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.trace(LinOpHelper(args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 1), (1, 4))
        expected = np.array([[1, 0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_mul(self, backend_name):
        """mul (left multiply) by constant matrix."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lhs = LinOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))
        out = backend.mul(LinOpHelper(data=lhs, args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_rmul(self, backend_name):
        """rmul (right multiply) by constant."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        rhs = LinOpHelper((2,), type="dense_const", data=np.array([1, 2]))
        out = backend.rmul(LinOpHelper(data=rhs, args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 2), (2, 4))
        expected = np.array([[1, 0, 2, 0], [0, 1, 0, 2]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_mul_elem(self, backend_name):
        """mul_elem (elementwise multiply) scales rows."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lhs = LinOpHelper((2,), type="dense_const", data=np.array([2, 3]))
        out = backend.mul_elem(LinOpHelper(data=lhs), view)
        A = to_dense(out.get_tensor_representation(0, 2), (2, 2))
        expected = np.array([[2, 0], [0, 3]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_div(self, backend_name):
        """div (elementwise divide) scales rows."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lhs = LinOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))
        out = backend.div(LinOpHelper(data=lhs), view)
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        expected = np.diag([1, 1/3, 1/2, 1/4])
        np.testing.assert_allclose(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_hstack(self, backend_name):
        """hstack concatenates variables horizontally."""
        backend = make_backend(backend_name, id_to_col={1: 0, 2: 1})
        x = LinOpHelper((1,), type="variable", data=1)
        y = LinOpHelper((1,), type="variable", data=2)
        out = backend.hstack(LinOpHelper(args=[x, y]), backend.get_empty_view())
        A = to_dense(out.get_tensor_representation(0, 2), (2, 2))
        np.testing.assert_array_equal(A, np.eye(2))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_vstack(self, backend_name):
        """vstack concatenates variables vertically."""
        backend = make_backend(backend_name, id_to_col={1: 0, 2: 2})
        x = LinOpHelper((1, 2), type="variable", data=1)
        y = LinOpHelper((1, 2), type="variable", data=2)
        out = backend.vstack(LinOpHelper(args=[x, y]), backend.get_empty_view())
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_concatenate_axis0(self, backend_name):
        """concatenate along axis=0 (like vstack)."""
        backend = make_backend(backend_name, id_to_col={1: 0, 2: 2})
        x = LinOpHelper((1, 2), type="variable", data=1)
        y = LinOpHelper((1, 2), type="variable", data=2)
        out = backend.concatenate(LinOpHelper(args=[x, y], data=[0]), backend.get_empty_view())
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_concatenate_axis1(self, backend_name):
        """concatenate along axis=1 (like hstack for matrices)."""
        backend = make_backend(backend_name, id_to_col={1: 0, 2: 2})
        x = LinOpHelper((1, 2), type="variable", data=1)
        y = LinOpHelper((1, 2), type="variable", data=2)
        out = backend.concatenate(LinOpHelper(args=[x, y], data=[1]), backend.get_empty_view())
        A = to_dense(out.get_tensor_representation(0, 4), (4, 4))
        np.testing.assert_array_equal(A, np.eye(4))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_conv(self, backend_name):
        """conv creates Toeplitz matrix."""
        backend = make_backend(backend_name)
        var = LinOpHelper((3,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        f = LinOpHelper((3,), type="dense_const", data=np.array([1, 2, 3]))
        out = backend.conv(LinOpHelper(data=f, shape=(5, 1), args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 5), (5, 3))
        expected = np.array([
            [1, 0, 0], [2, 1, 0], [3, 2, 1], [0, 3, 2], [0, 0, 3]
        ], dtype=float)
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_kron_r(self, backend_name):
        """kron_r computes kron(const, x)."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        a = LinOpHelper((2, 1), type="dense_const", data=np.array([[1], [2]]))
        out = backend.kron_r(LinOpHelper(data=a, args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 8), (8, 4))
        expected = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 2, 0], [0, 0, 0, 2],
        ], dtype=float)
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_kron_l(self, backend_name):
        """kron_l computes kron(x, const)."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        a = LinOpHelper((1, 2), type="dense_const", data=np.array([[1, 2]]))
        out = backend.kron_l(LinOpHelper(data=a, args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 8), (8, 4))
        expected = np.array([
            [1, 0, 0, 0], [0, 1, 0, 0], [2, 0, 0, 0], [0, 2, 0, 0],
            [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 2, 0], [0, 0, 0, 2],
        ], dtype=float)
        np.testing.assert_array_equal(A, expected)


class TestBackendConsistency:
    """Verify all backends produce identical outputs for all operations."""

    @pytest.mark.parametrize("linop_name", LINOPS)
    def test_linop_consistency(self, linop_name):
        """All backends should produce same output for each linop."""
        results = {}
        for backend_name in BACKENDS:
            backend = self._make_backend_for_linop(backend_name, linop_name)
            try:
                result = self._apply_linop(backend, linop_name)
                results[backend_name] = result
            except Exception as e:
                pytest.fail(f"Backend {backend_name} failed on {linop_name}: {e}")

        # Compare all backends to first
        ref_name = BACKENDS[0]
        ref = results[ref_name]
        for name, arr in results.items():
            if name != ref_name:
                np.testing.assert_allclose(
                    arr, ref, rtol=1e-12, atol=1e-12,
                    err_msg=f"{name} differs from {ref_name} on {linop_name}"
                )

    def _make_backend_for_linop(self, backend_name, linop_name):
        """Create backend with appropriate configuration for each linop."""
        if linop_name in ("hstack", "vstack", "concatenate"):
            return make_backend(backend_name, id_to_col={1: 0, 2: 2})
        return make_backend(backend_name)

    def _apply_linop(self, backend, linop_name: str) -> np.ndarray:
        """Apply a linop and return dense array result."""
        if linop_name == "neg":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.neg(LinOpHelper(), view)
            return to_dense(out.get_tensor_representation(0, 4), (4, 4))

        elif linop_name == "sum_entries":
            var = LinOpHelper((3,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lin_op = LinOpHelper(shape=(3,), data=[None, True], args=[var])
            out = backend.sum_entries(lin_op, view)
            return to_dense(out.get_tensor_representation(0, 1), (1, 3))

        elif linop_name == "mul":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lhs = LinOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))
            out = backend.mul(LinOpHelper(data=lhs, args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 4), (4, 4))

        elif linop_name == "rmul":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            rhs = LinOpHelper((2,), type="dense_const", data=np.array([1, 2]))
            out = backend.rmul(LinOpHelper(data=rhs, args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 2), (2, 4))

        elif linop_name == "mul_elem":
            var = LinOpHelper((2,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lhs = LinOpHelper((2,), type="dense_const", data=np.array([2, 3]))
            out = backend.mul_elem(LinOpHelper(data=lhs), view)
            return to_dense(out.get_tensor_representation(0, 2), (2, 2))

        elif linop_name == "div":
            var = LinOpHelper((2,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lhs = LinOpHelper((2,), type="dense_const", data=np.array([2, 4]))
            out = backend.div(LinOpHelper(data=lhs), view)
            return to_dense(out.get_tensor_representation(0, 2), (2, 2))

        elif linop_name == "promote":
            var = LinOpHelper((1,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.promote(LinOpHelper((3,)), view)
            return to_dense(out.get_tensor_representation(0, 1), (3, 1))

        elif linop_name == "broadcast_to":
            var = LinOpHelper((3,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lin_op = LinOpHelper((2, 3), data=(2, 3), args=[var])
            out = backend.broadcast_to(lin_op, view)
            return to_dense(out.get_tensor_representation(0, 3), (6, 3))

        elif linop_name == "reshape":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.reshape(LinOpHelper((4,), args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 4), (4, 4))

        elif linop_name == "transpose":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.transpose(LinOpHelper((2, 2), data=[None], args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 4), (4, 4))

        elif linop_name == "index":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lin_op = LinOpHelper(data=[slice(0, 2, 1), slice(0, 1, 1)], args=[var])
            out = backend.index(lin_op, view)
            return to_dense(out.get_tensor_representation(0, 2), (2, 4))

        elif linop_name == "diag_vec":
            var = LinOpHelper((2,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.diag_vec(LinOpHelper(shape=(2, 2), data=0), view)
            return to_dense(out.get_tensor_representation(0, 4), (4, 2))

        elif linop_name == "diag_mat":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.diag_mat(LinOpHelper(shape=(2, 2), data=0), view)
            return to_dense(out.get_tensor_representation(0, 2), (2, 4))

        elif linop_name == "trace":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.trace(LinOpHelper(args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 1), (1, 4))

        elif linop_name == "upper_tri":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.upper_tri(LinOpHelper(args=[LinOpHelper((2, 2))]), view)
            return to_dense(out.get_tensor_representation(0, 1), (1, 4))

        elif linop_name == "hstack":
            x = LinOpHelper((1,), type="variable", data=1)
            y = LinOpHelper((1,), type="variable", data=2)
            out = backend.hstack(LinOpHelper(args=[x, y]), backend.get_empty_view())
            return to_dense(out.get_tensor_representation(0, 2), (2, 4))

        elif linop_name == "vstack":
            x = LinOpHelper((1, 2), type="variable", data=1)
            y = LinOpHelper((1, 2), type="variable", data=2)
            out = backend.vstack(LinOpHelper(args=[x, y]), backend.get_empty_view())
            return to_dense(out.get_tensor_representation(0, 4), (4, 4))

        elif linop_name == "concatenate":
            x = LinOpHelper((1, 2), type="variable", data=1)
            y = LinOpHelper((1, 2), type="variable", data=2)
            out = backend.concatenate(LinOpHelper(args=[x, y], data=[0]), backend.get_empty_view())
            return to_dense(out.get_tensor_representation(0, 4), (4, 4))

        elif linop_name == "conv":
            var = LinOpHelper((3,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            f = LinOpHelper((3,), type="dense_const", data=np.array([1, 2, 3]))
            out = backend.conv(LinOpHelper(data=f, shape=(5, 1), args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 5), (5, 3))

        elif linop_name == "kron_l":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            a = LinOpHelper((1, 2), type="dense_const", data=np.array([[1, 2]]))
            out = backend.kron_l(LinOpHelper(data=a, args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 8), (8, 4))

        elif linop_name == "kron_r":
            var = LinOpHelper((2, 2), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            a = LinOpHelper((2, 1), type="dense_const", data=np.array([[1], [2]]))
            out = backend.kron_r(LinOpHelper(data=a, args=[var]), view)
            return to_dense(out.get_tensor_representation(0, 8), (8, 4))

        else:
            raise ValueError(f"Unknown linop: {linop_name}")


class TestEdgeCases:
    """Test edge cases: scalars, higher dimensions, etc."""

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_scalar_variable(self, backend_name):
        """Operations on scalar (1,) variables."""
        backend = make_backend(backend_name)
        var = LinOpHelper((1,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # neg on scalar
        out = backend.neg(LinOpHelper(), view)
        A = to_dense(out.get_tensor_representation(0, 1), (1, 1))
        np.testing.assert_array_equal(A, [[-1]])

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_3d_variable_reshape(self, backend_name):
        """Reshape on 3D variable."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # Reshape to flat
        out = backend.reshape(LinOpHelper((8,), args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, 8), (8, 8))
        np.testing.assert_array_equal(A, np.eye(8))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_3d_concatenate(self, backend_name):
        """Concatenate 3D variables."""
        backend = make_backend(backend_name, id_to_col={1: 0, 2: 8})
        x = LinOpHelper((2, 2, 2), type="variable", data=1)
        y = LinOpHelper((2, 2, 2), type="variable", data=2)

        # Concatenate along axis 2
        out = backend.concatenate(LinOpHelper(args=[x, y], data=[2]), backend.get_empty_view())
        A = to_dense(out.get_tensor_representation(0, 16), (16, 16))
        np.testing.assert_array_equal(A, np.eye(16))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_diag_vec_offset(self, backend_name):
        """diag_vec with k offset."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # k=1 means superdiagonal
        out = backend.diag_vec(LinOpHelper(shape=(3, 3), data=1), view)
        A = to_dense(out.get_tensor_representation(0, 9), (9, 2))
        # [0, x1, 0; 0, 0, x2; 0, 0, 0] in column-major = [0,0,0,x1,0,0,0,x2,0]
        expected = np.zeros((9, 2))
        expected[3, 0] = 1
        expected[7, 1] = 1
        np.testing.assert_array_equal(A, expected)

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_diag_mat_offset(self, backend_name):
        """diag_mat with k offset."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # k=1 extracts superdiagonal
        out = backend.diag_mat(LinOpHelper(shape=(1, 1), data=1), view)
        A = to_dense(out.get_tensor_representation(0, 1), (1, 4))
        expected = np.array([[0, 0, 1, 0]])
        np.testing.assert_array_equal(A, expected)


class TestBuildMatrix:
    """Test the full build_matrix pipeline."""

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_simple_constraint(self, backend_name):
        """Build matrix for simple variable constraint."""
        backend = make_backend(backend_name)

        # Create a simple variable LinOp tree
        var = LinOpHelper((2,), type="variable", data=1)
        result = backend.build_matrix([var])

        # Result should be (var_length+1, param_size_plus_one) shape flattened
        # For var_length=8, param_size_plus_one=2
        assert result.shape == (2 * (8 + 1), 2)


def to_dense_param(
        tr: TensorRepresentation, 
        shape: tuple[int, int], 
        param_offset: int
    ) -> np.ndarray:
    """Convert TensorRepresentation to dense array for a specific param offset."""
    mask = tr.parameter_offset == param_offset
    return sp.coo_matrix(
        (tr.data[mask], (tr.row[mask], tr.col[mask])), shape=shape
    ).toarray()


class TestParametrizedLinops:
    """Test linops with parameters - catches DPP tensor issues."""

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_mul_param(self, backend_name):
        """Left multiply variable by parameter: param @ x."""
        # param_id=3 has size 4 (2x2 matrix), maps to col offset 1
        backend = make_backend(backend_name)
        var = LinOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # Create param LinOp (2x2 @ 2x1 -> 2x1)
        param = LinOpHelper((2, 2), type="param", data=3)
        lin_op = LinOpHelper(shape=(2,), data=param, args=[var])
        out = backend.mul(lin_op, view)

        tr = out.get_tensor_representation(0, 2)

        # Should have entries for param slices (offset 1-4)
        param_offsets = set(tr.parameter_offset)
        assert len(param_offsets) > 1, "Should have multiple param offsets"
        assert 0 not in param_offsets or len(param_offsets) > 1, "Should have param entries"

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_rmul_param(self, backend_name):
        """Right multiply variable by parameter: x @ param (dotsort pattern)."""
        # Use param_id=3 with size=2 for a (2,) param vector
        backend = make_backend(
            backend_name,
            param_to_size={-1: 1, 3: 2},
            param_to_col={-1: 0, 3: 1},
            param_size_plus_one=3,
        )
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # x (2x2) @ param (2,) -> (2,)
        param = LinOpHelper((2,), type="param", data=3)
        lin_op = LinOpHelper(shape=(2,), data=param, args=[var])
        out = backend.rmul(lin_op, view)

        tr = out.get_tensor_representation(0, 2)

        # Should have param entries - one offset per param element
        param_offsets = set(tr.parameter_offset)
        assert len(param_offsets) == 2, f"Expected 2 param offsets, got {param_offsets}"

    @pytest.mark.parametrize("backend_name", BACKENDS)
    def test_mul_elem_param(self, backend_name):
        """Elementwise multiply by parameter."""
        backend = make_backend(backend_name)
        var = LinOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # param (2x2) * x (2x2) elementwise
        param = LinOpHelper((2, 2), type="param", data=3)
        lin_op = LinOpHelper(shape=(2, 2), data=param, args=[var])
        out = backend.mul_elem(lin_op, view)

        tr = out.get_tensor_representation(0, 4)

        # Should have param entries
        param_offsets = set(tr.parameter_offset)
        assert len(param_offsets) > 1, "Should have multiple param offsets"

    def test_param_consistency(self):
        """All backends produce identical results for parametrized mul."""
        results = {}
        for backend_name in BACKENDS:
            backend = make_backend(backend_name)
            var = LinOpHelper((2,), type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())

            param = LinOpHelper((2, 2), type="param", data=3)
            lin_op = LinOpHelper(shape=(2,), data=param, args=[var])
            out = backend.mul(lin_op, view)

            tr = out.get_tensor_representation(0, 2)
            # Store dense arrays for each param offset
            results[backend_name] = {
                p: to_dense_param(tr, (2, 2), p)
                for p in sorted(set(tr.parameter_offset))
            }

        # Compare all to first
        ref_name = BACKENDS[0]
        ref = results[ref_name]
        for name, res in results.items():
            if name != ref_name:
                assert set(res.keys()) == set(ref.keys()), \
                    f"{name} has different param offsets than {ref_name}"
                for p in ref.keys():
                    np.testing.assert_allclose(
                        res[p], ref[p], rtol=1e-12, atol=1e-12,
                        err_msg=f"{name} differs from {ref_name} at param_offset={p}"
                    )


# N-dimensional test shapes (3D and 4D)
ND_SHAPES = [
    (2, 3, 4),      # 3D: 24 elements
    (3, 2, 2),      # 3D: 12 elements
    (2, 2, 2, 2),   # 4D: 16 elements
    (2, 3, 2, 2),   # 4D: 24 elements
]


class TestNDimensional:
    """Test linops with 3D and 4D arrays."""

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_variable_nd(self, backend_name, shape):
        """Variable creation with ND shapes."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        tr = view.get_tensor_representation(0, size)
        A = to_dense(tr, (size, size))
        np.testing.assert_array_equal(A, np.eye(size))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_neg_nd(self, backend_name, shape):
        """Negation on ND variable."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.neg(LinOpHelper(), view)
        A = to_dense(out.get_tensor_representation(0, size), (size, size))
        np.testing.assert_array_equal(A, -np.eye(size))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_reshape_nd_to_1d(self, backend_name, shape):
        """Reshape ND to flat 1D."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.reshape(LinOpHelper((size,), args=[var]), view)
        A = to_dense(out.get_tensor_representation(0, size), (size, size))
        np.testing.assert_array_equal(A, np.eye(size))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_sum_entries_nd(self, backend_name, shape):
        """Sum all entries of ND variable."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        lin_op = LinOpHelper(shape=shape, data=[None, True], args=[var])
        out = backend.sum_entries(lin_op, view)
        A = to_dense(out.get_tensor_representation(0, 1), (1, size))
        np.testing.assert_array_equal(A, np.ones((1, size)))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_promote_to_nd(self, backend_name, shape):
        """Promote scalar to ND shape."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=1)
        var = LinOpHelper((1,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        out = backend.promote(LinOpHelper(shape), view)
        A = to_dense(out.get_tensor_representation(0, 1), (size, 1))
        np.testing.assert_array_equal(A, np.ones((size, 1)))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", [(2, 3, 4), (3, 2, 2)])
    def test_index_nd(self, backend_name, shape):
        """Index into ND variable."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        # Index first element along each dimension
        slices = [slice(0, 1, 1) for _ in shape]
        lin_op = LinOpHelper(data=slices, args=[var])
        out = backend.index(lin_op, view)
        tr = out.get_tensor_representation(0, 1)
        # Should select exactly 1 element (check non-zero count)
        A = to_dense(tr, (1, size))
        assert np.count_nonzero(A) == 1
        assert A[0, 0] == 1.0  # First element selected

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_mul_scalar_nd(self, backend_name, shape):
        """Multiply ND variable by scalar constant."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        # Scalar multiply
        lhs = LinOpHelper((1,), type="dense_const", data=np.array([2.0]))
        out = backend.mul_elem(LinOpHelper(data=lhs), view)
        A = to_dense(out.get_tensor_representation(0, size), (size, size))
        np.testing.assert_array_equal(A, 2.0 * np.eye(size))

    @pytest.mark.parametrize("backend_name", BACKENDS)
    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_div_scalar_nd(self, backend_name, shape):
        """Divide ND variable by scalar constant."""
        size = int(np.prod(shape))
        backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
        var = LinOpHelper(shape, type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())
        # Scalar divide (element-wise by 2)
        lhs = LinOpHelper(shape, type="dense_const", data=2.0 * np.ones(shape))
        out = backend.div(LinOpHelper(data=lhs), view)
        A = to_dense(out.get_tensor_representation(0, size), (size, size))
        np.testing.assert_allclose(A, 0.5 * np.eye(size))


class TestNDConsistency:
    """Cross-backend consistency tests for ND arrays."""

    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_nd_variable_consistency(self, shape):
        """All backends produce same result for ND variable."""
        size = int(np.prod(shape))
        results = {}
        for backend_name in BACKENDS:
            backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
            var = LinOpHelper(shape, type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            tr = view.get_tensor_representation(0, size)
            results[backend_name] = to_dense(tr, (size, size))

        ref_name = BACKENDS[0]
        for name, arr in results.items():
            if name != ref_name:
                np.testing.assert_allclose(
                    arr, results[ref_name], rtol=1e-12,
                    err_msg=f"{name} differs from {ref_name} for shape {shape}"
                )

    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_nd_reshape_consistency(self, shape):
        """All backends produce same result for ND reshape."""
        size = int(np.prod(shape))
        results = {}
        for backend_name in BACKENDS:
            backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
            var = LinOpHelper(shape, type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            out = backend.reshape(LinOpHelper((size,), args=[var]), view)
            tr = out.get_tensor_representation(0, size)
            results[backend_name] = to_dense(tr, (size, size))

        ref_name = BACKENDS[0]
        for name, arr in results.items():
            if name != ref_name:
                np.testing.assert_allclose(
                    arr, results[ref_name], rtol=1e-12,
                    err_msg=f"{name} differs from {ref_name} for reshape {shape}"
                )

    @pytest.mark.parametrize("shape", ND_SHAPES)
    def test_nd_sum_consistency(self, shape):
        """All backends produce same result for ND sum_entries."""
        size = int(np.prod(shape))
        results = {}
        for backend_name in BACKENDS:
            backend = make_backend(backend_name, id_to_col={1: 0}, var_length=size)
            var = LinOpHelper(shape, type="variable", data=1)
            view = backend.process_constraint(var, backend.get_empty_view())
            lin_op = LinOpHelper(shape=shape, data=[None, True], args=[var])
            out = backend.sum_entries(lin_op, view)
            tr = out.get_tensor_representation(0, 1)
            results[backend_name] = to_dense(tr, (1, size))

        ref_name = BACKENDS[0]
        for name, arr in results.items():
            if name != ref_name:
                np.testing.assert_allclose(
                    arr, results[ref_name], rtol=1e-12,
                    err_msg=f"{name} differs from {ref_name} for sum {shape}"
                )
