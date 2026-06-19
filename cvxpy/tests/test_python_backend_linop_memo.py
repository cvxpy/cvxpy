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

import cvxpy.lin_ops.backends.coo_backend as coo_backend
import cvxpy.settings as s
from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops import lin_utils as lu
from cvxpy.lin_ops.backends.coo_backend import CooCanonBackend
from cvxpy.lin_ops.backends.scipy_backend import SciPyCanonBackend

PYTHON_BACKENDS = [
    (s.SCIPY_CANON_BACKEND, SciPyCanonBackend),
    (s.COO_CANON_BACKEND, CooCanonBackend),
]


def _problem_matrix(lin_ops, backend):
    return canonInterface.get_problem_matrix(
        lin_ops,
        var_length=3,
        id_to_col={1: 0},
        param_to_size={-1: 1},
        param_to_col={-1: 0},
        constr_length=sum(np.prod(lin_op.shape) for lin_op in lin_ops),
        canon_backend=backend,
    )


def _problem_matrix_with_var_length(lin_ops, backend, var_length, param_to_size=None):
    param_to_size = {-1: 1} if param_to_size is None else param_to_size
    param_to_col = {}
    offset = 0
    for param_id, size in param_to_size.items():
        param_to_col[param_id] = offset
        offset += size
    return canonInterface.get_problem_matrix(
        lin_ops,
        var_length=var_length,
        id_to_col={1: 0},
        param_to_size=param_to_size,
        param_to_col=param_to_col,
        constr_length=sum(np.prod(lin_op.shape) for lin_op in lin_ops),
        canon_backend=backend,
    )


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_shared_linop_lowered_once(monkeypatch, backend, backend_cls) -> None:
    x = lu.create_var((3,), var_id=1)
    calls = 0
    original = backend_cls.get_variable_tensor

    def spy_get_variable_tensor(self, shape, variable_id):
        nonlocal calls
        calls += 1
        return original(self, shape, variable_id)

    monkeypatch.setattr(backend_cls, "get_variable_tensor", spy_get_variable_tensor)

    matrix = _problem_matrix([x, x], backend)

    assert calls == 1
    assert matrix.shape == (2 * 3 * (3 + 1), 1)
    assert matrix.nnz == 2 * 3


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_single_use_linop_is_not_cached(monkeypatch, backend, backend_cls) -> None:
    x = lu.create_var((3,), var_id=1)
    calls = 0
    original = backend_cls._copy_tensor_view

    def spy_copy_tensor_view(self, view, empty_view):
        nonlocal calls
        calls += 1
        return original(self, view, empty_view)

    monkeypatch.setattr(backend_cls, "_copy_tensor_view", spy_copy_tensor_view)

    matrix = _problem_matrix([x], backend)

    assert calls == 0
    assert matrix.shape == (3 * (3 + 1), 1)
    assert matrix.nnz == 3


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_descendant_of_reused_linop_is_not_cached(monkeypatch, backend, backend_cls) -> None:
    x = lu.create_var((3,), var_id=1)
    neg_x = lu.neg_expr(x)
    calls = 0
    original = backend_cls._copy_tensor_view

    def spy_copy_tensor_view(self, view, empty_view):
        nonlocal calls
        calls += 1
        return original(self, view, empty_view)

    monkeypatch.setattr(backend_cls, "_copy_tensor_view", spy_copy_tensor_view)

    matrix = _problem_matrix([neg_x, neg_x], backend)

    assert calls == 2
    assert matrix.shape == (2 * 3 * (3 + 1), 1)
    assert matrix.nnz == 2 * 3


@pytest.mark.parametrize("backend", [backend for backend, _ in PYTHON_BACKENDS])
def test_shared_linop_cache_hits_do_not_share_mutable_views(backend) -> None:
    x = lu.create_var((3,), var_id=1)
    matrix = _problem_matrix([lu.neg_expr(x), x], backend).toarray()
    matrix = matrix.reshape((6, 4), order="F")

    first_block = matrix[:3, :3]
    second_block = matrix[3:, :3]

    np.testing.assert_array_equal(first_block, -np.eye(3))
    np.testing.assert_array_equal(second_block, np.eye(3))


@pytest.mark.parametrize("backend", [backend for backend, _ in PYTHON_BACKENDS])
def test_trace_of_matmul_linop_coefficients(backend) -> None:
    n = 3
    x = lu.create_var((n, n), var_id=1)
    a = np.arange(1, n * n + 1, dtype=float).reshape((n, n), order="F")
    trace_op = lu.trace(lu.mul_expr(lu.create_const(a, (n, n)), x, (n, n)))

    matrix = _problem_matrix_with_var_length([trace_op], backend, n * n).toarray()

    coeffs = matrix[:n * n, 0]
    np.testing.assert_array_equal(coeffs, a.T.flatten(order="F"))


def test_coo_trace_of_matmul_uses_restricted_rows(monkeypatch) -> None:
    n = 4
    x = lu.create_var((n, n), var_id=1)
    a = lu.create_const(np.ones((n, n)), (n, n))
    trace_op = lu.trace(lu.mul_expr(a, x, (n, n)))
    calls = {"full": 0, "restricted": []}
    original_full = coo_backend._kron_nd_structure_mul
    original_restricted = coo_backend._kron_nd_structure_mul_rows

    def spy_full(*args, **kwargs):
        calls["full"] += 1
        return original_full(*args, **kwargs)

    def spy_restricted(tensor, batch_size, cols, rows):
        calls["restricted"].append(np.asarray(rows).copy())
        return original_restricted(tensor, batch_size, cols, rows)

    monkeypatch.setattr(coo_backend, "_kron_nd_structure_mul", spy_full)
    monkeypatch.setattr(coo_backend, "_kron_nd_structure_mul_rows", spy_restricted)

    matrix = _problem_matrix_with_var_length([trace_op], s.COO_CANON_BACKEND, n * n)

    assert matrix.nnz == n * n
    assert calls["full"] == 0
    assert len(calls["restricted"]) == 1
    np.testing.assert_array_equal(calls["restricted"][0], np.arange(n) * (n + 1))


def test_coo_index_through_hstack_uses_restricted_parametric_mul(monkeypatch) -> None:
    n = 5
    x = lu.create_var((n,), var_id=1)
    p = lu.create_param((n, n), param_id=2)
    px = lu.mul_expr(p, x, (n,))
    stacked = lu.hstack([px, x], (2 * n,))
    indexed = lu.index(stacked, (1,), (slice(2, 3, 1),))
    calls = []
    original_restricted = coo_backend._kron_nd_structure_mul_rows

    def spy_restricted(tensor, batch_size, cols, rows):
        calls.append(np.asarray(rows).copy())
        return original_restricted(tensor, batch_size, cols, rows)

    monkeypatch.setattr(coo_backend, "_kron_nd_structure_mul_rows", spy_restricted)

    matrix = _problem_matrix_with_var_length(
        [indexed],
        s.COO_CANON_BACKEND,
        n,
        param_to_size={-1: 1, 2: n * n},
    )

    assert matrix.nnz == n
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.array([2]))
