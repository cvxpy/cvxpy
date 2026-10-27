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
import cvxpy.settings as s
from cvxpy.cvxcore.python import canonInterface
from cvxpy.lin_ops import lin_utils as lu
from cvxpy.lin_ops.backends.coo_backend import CooCanonBackend
from cvxpy.lin_ops.backends.scipy_backend import SciPyCanonBackend

PYTHON_BACKENDS = [
    (s.SCIPY_CANON_BACKEND, SciPyCanonBackend),
    (s.COO_CANON_BACKEND, CooCanonBackend),
]


def _problem_matrix(lin_ops, backend, variables=None, parameters=None):
    if variables is None:
        var_length = 3
        id_to_col = {1: 0}
    else:
        var_length = 0
        id_to_col = {}
        for variable in variables:
            id_to_col[variable.id] = var_length
            var_length += int(np.prod(variable.shape))

    param_to_size = {-1: 1}
    param_to_col = {-1: 0}
    if parameters is not None:
        offset = 0
        param_to_size = {}
        param_to_col = {}
        for parameter in parameters:
            param_size = int(np.prod(parameter.shape))
            param_to_size[parameter.id] = param_size
            param_to_col[parameter.id] = offset
            offset += param_size
        param_to_size[-1] = 1
        param_to_col[-1] = offset

    return canonInterface.get_problem_matrix(
        lin_ops,
        var_length=var_length,
        id_to_col=id_to_col,
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

    shared = lu.neg_expr(x)
    matrix = _problem_matrix([shared, shared], backend)

    assert calls == 1
    assert matrix.shape == (2 * 3 * (3 + 1), 1)
    assert matrix.nnz == 2 * 3


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_shared_parameter_linop_in_data_is_lowered_once(monkeypatch, backend, backend_cls) -> None:
    P = cp.Parameter((3, 3), name="P")
    x = cp.Variable(3, name="x")
    y = cp.Variable(3, name="y")
    scaled = 2 * P
    lin_op_x, _ = (scaled @ x).canonical_form
    lin_op_y, _ = (scaled @ y).canonical_form

    assert lin_op_x.data is lin_op_y.data
    assert lin_op_x.data.type == "mul_elem"

    calls = 0
    original = backend_cls.get_param_tensor

    def spy_get_param_tensor(self, shape, parameter_id):
        nonlocal calls
        calls += 1
        return original(self, shape, parameter_id)

    monkeypatch.setattr(backend_cls, "get_param_tensor", spy_get_param_tensor)

    matrix = _problem_matrix([lin_op_x, lin_op_y], backend, variables=[x, y], parameters=[P])

    assert calls == 1
    assert matrix.shape == (2 * 3 * (2 * 3 + 1), 3 * 3 + 1)
    assert matrix.nnz == 2 * 3 * 3
    np.testing.assert_array_equal(matrix.data, 2 * np.ones(matrix.nnz))


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

    matrix = _problem_matrix([lu.neg_expr(x)], backend)

    assert calls == 0
    assert matrix.shape == (3 * (3 + 1), 1)
    assert matrix.nnz == 3


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_descendant_of_reused_linop_is_not_cached(monkeypatch, backend, backend_cls) -> None:
    x = lu.create_var((3,), var_id=1)
    descendant = lu.neg_expr(x)
    shared = lu.neg_expr(descendant)
    calls = 0
    original = backend_cls._copy_tensor_view

    def spy_copy_tensor_view(self, view, empty_view):
        nonlocal calls
        calls += 1
        return original(self, view, empty_view)

    monkeypatch.setattr(backend_cls, "_copy_tensor_view", spy_copy_tensor_view)

    matrix = _problem_matrix([shared, shared], backend)

    assert calls == 1
    assert matrix.shape == (2 * 3 * (3 + 1), 1)
    assert matrix.nnz == 2 * 3


@pytest.mark.parametrize("backend", [backend for backend, _ in PYTHON_BACKENDS])
def test_shared_linop_cache_hits_do_not_share_mutable_views(backend) -> None:
    x = lu.create_var((3,), var_id=1)
    shared = lu.sum_expr([x, x])
    matrix = _problem_matrix([lu.neg_expr(shared), lu.neg_expr(shared), shared], backend).toarray()
    matrix = matrix.reshape((9, 4), order="F")

    first_block = matrix[:3, :3]
    second_block = matrix[3:6, :3]
    third_block = matrix[6:, :3]

    np.testing.assert_array_equal(first_block, -2 * np.eye(3))
    np.testing.assert_array_equal(second_block, -2 * np.eye(3))
    np.testing.assert_array_equal(third_block, 2 * np.eye(3))


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_nested_shared_graph_is_counted_once(monkeypatch, backend, backend_cls) -> None:
    shared = lu.create_var((3,), var_id=1)
    nodes = [shared]
    for _ in range(40):
        shared = lu.sum_expr([shared, shared])
        nodes.append(shared)

    expanded = set()
    original = backend_cls._lin_op_children

    def spy_children(lin_op):
        # Fail immediately on repeated expansion instead of timing an exponential walk.
        assert id(lin_op) not in expanded
        expanded.add(id(lin_op))
        return original(lin_op)

    monkeypatch.setattr(backend_cls, "_lin_op_children", staticmethod(spy_children))
    instance = backend_cls({1: 0}, {-1: 1}, {-1: 0}, 1, 3)
    counts = instance._count_reusable_lin_ops([shared])

    assert expanded == {id(node) for node in nodes}
    assert counts[id(shared)] == 1
    assert all(counts[id(node)] == 2 for node in nodes[:-1])


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
@pytest.mark.parametrize("leaf_type", ["variable", "param", "scalar_const", "dense_const",
                                       "sparse_const"])
def test_repeated_leaves_are_not_cached(monkeypatch, backend, backend_cls, leaf_type) -> None:
    parameter = cp.Parameter(3)
    if leaf_type == "variable":
        leaf = lu.create_var((3,), var_id=1)
    elif leaf_type == "param":
        leaf, _ = parameter.canonical_form
    elif leaf_type == "scalar_const":
        leaf = lu.create_const(2, (1, 1))
    elif leaf_type == "dense_const":
        leaf = lu.create_const(np.ones(3), (3,))
    else:
        leaf = lu.create_const(sp.eye_array(3), (3, 3), sparse=True)
    assert leaf.type == leaf_type

    def fail_copy(*args):
        pytest.fail("Leaf nodes must not enter the tensor cache")

    monkeypatch.setattr(backend_cls, "_copy_tensor_view", fail_copy)
    _problem_matrix([leaf, leaf], backend, parameters=[parameter])


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_last_use_transfers_cached_view(monkeypatch, backend, backend_cls) -> None:
    shared = lu.neg_expr(lu.create_var((3,), var_id=1))
    original = backend_cls.process_constraint
    snapshots = []

    def spy_process(self, lin_op, empty_view):
        cached = self._lin_op_tensor_cache.get(id(shared))
        result = original(self, lin_op, empty_view)
        if lin_op is shared:
            snapshots.append((cached, result, id(shared) in self._lin_op_tensor_cache))
        return result

    monkeypatch.setattr(backend_cls, "process_constraint", spy_process)
    _problem_matrix([shared, shared, shared], backend)

    assert [present for _, _, present in snapshots] == [True, True, False]
    cached, result, _ = snapshots[1]
    assert result is not cached
    assert snapshots[2][1] is cached


@pytest.mark.parametrize(("backend", "backend_cls"), PYTHON_BACKENDS)
def test_tensor_copy_shares_arrays_but_not_wrappers(backend, backend_cls) -> None:
    instance = backend_cls({1: 0}, {-1: 1}, {-1: 0}, 1, 3)
    tensor = instance.get_variable_tensor((3,), 1)
    copied = instance._copy_tensor_data(tensor)

    assert copied is not tensor
    original_matrix = tensor[1][-1]
    copied_matrix = copied[1][-1]
    assert copied[1] is not tensor[1]
    assert copied_matrix is not original_matrix
    fields = ("data", "indices", "indptr") if sp.issparse(original_matrix) else (
        "data", "row", "col", "param_idx")
    for field in fields:
        assert getattr(copied_matrix, field) is getattr(original_matrix, field)
    copied_matrix.data = -copied_matrix.data
    np.testing.assert_array_equal(original_matrix.data, np.ones(3))


@pytest.mark.parametrize("backend", [backend for backend, _ in PYTHON_BACKENDS])
def test_shared_constant_data_survives_division(backend) -> None:
    x = lu.create_var((3,), var_id=1)
    shared = lu.neg_expr(lu.create_const(-2 * np.ones(3), (3,)))
    matrix = _problem_matrix([lu.div_expr(x, shared), lu.div_expr(x, shared), shared],
                             backend).toarray().reshape((9, 4), order="F")
    np.testing.assert_array_equal(matrix[:3, :3], np.eye(3) / 2)
    np.testing.assert_array_equal(matrix[3:6, :3], np.eye(3) / 2)
    np.testing.assert_array_equal(matrix[6:, 3], 2 * np.ones(3))
