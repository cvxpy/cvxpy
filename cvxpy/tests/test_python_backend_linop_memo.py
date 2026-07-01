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
