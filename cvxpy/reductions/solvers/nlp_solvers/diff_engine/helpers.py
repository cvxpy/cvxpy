"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared helpers for converting CVXPY expressions to C diff engine expressions.
"""
import numpy as np
from scipy import sparse
from sparsediffpy import _sparsediffengine as _diffengine


def normalize_shape(shape):
    """Normalize shape to 2D (d1, d2) for the C engine."""
    shape = tuple(shape)
    return (1,) * (2 - len(shape)) + shape


def to_dense_float(value):
    """Convert a value to a dense float64 numpy array."""
    if sparse.issparse(value):
        value = value.todense()
    return np.asarray(value, dtype=np.float64)


def chain_add(children):
    """Chain multiple children with binary adds: a + b + c -> add(add(a, b), c)."""
    result = children[0]
    for child in children[1:]:
        result = _diffengine.make_add(result, child)
    return result


def make_sparse_left_matmul(param_node, child, A):
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    return _diffengine.make_left_matmul(
        param_node, child, 'sparse',
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int32, copy=False),
        A.indptr.astype(np.int32, copy=False),
        A.shape[0], A.shape[1])


def make_dense_left_matmul(param_node, child, A):
    m, n = normalize_shape(A.shape)
    return _diffengine.make_left_matmul(
        param_node, child, 'dense', A.flatten(order='C'), m, n)


def make_sparse_right_matmul(param_node, child, A):
    if not isinstance(A, sparse.csr_matrix):
        A = sparse.csr_matrix(A)
    return _diffengine.make_right_matmul(
        param_node, child, 'sparse',
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int32, copy=False),
        A.indptr.astype(np.int32, copy=False),
        A.shape[0], A.shape[1])


def make_dense_right_matmul(param_node, child, A):
    m, n = normalize_shape(A.shape)
    return _diffengine.make_right_matmul(
        param_node, child, 'dense', A.flatten(order='C'), m, n)


def build_var_dict(inverse_data):
    """Build {var_id: C variable capsule} mapping from InverseData."""
    n_vars = inverse_data.x_length
    var_dict = {}
    for var_id, (offset, _) in inverse_data.id_map.items():
        d1, d2 = normalize_shape(inverse_data.var_shapes[var_id])
        var_dict[var_id] = _diffengine.make_variable(d1, d2, offset, n_vars)
    return var_dict, n_vars


def build_param_dict(problem, inverse_data):
    """Build {param_id: C parameter capsule} mapping from InverseData."""
    n_vars = inverse_data.x_length
    param_dict = {}
    for param_id, offset in inverse_data.param_id_map.items():
        # this is needed to not get key errors with Constants.
        if param_id not in inverse_data.param_shapes:
            continue
        d1, d2 = normalize_shape(inverse_data.param_shapes[param_id])
        # TODO this is a bit hacky, potentially we can just store the initial
        # values in the InverseData, but we need to discuss with others.
        param = next(p for p in problem.parameters() if p.id == param_id)
        p = to_dense_float(param.value)
        param_dict[param_id] = _diffengine.make_parameter(
            d1, d2, offset, n_vars, p.flatten(order='F'))
    return param_dict
