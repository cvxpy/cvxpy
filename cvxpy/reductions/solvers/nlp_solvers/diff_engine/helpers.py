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

import cvxpy.settings as s


def normalize_shape(shape):
    """Normalize shape to 2D (d1, d2) for the C engine."""
    shape = tuple(shape)
    if len(shape) > 2:
        raise NotImplementedError(
            f">2-D expressions (shape {shape}) are not supported by the diff "
            "engine; the engine represents all expressions as 2-D matrices."
        )
    return (1,) * (2 - len(shape)) + shape


def to_dense_float(value):
    """Convert a value to a dense float64 numpy array."""
    if sparse.issparse(value):
        value = value.toarray()
    return np.asarray(value, dtype=np.float64)


def chain_add(children):
    """Combine children with a balanced binary tree of adds.

    Tree depth is ceil(log2(N)) instead of N-1, which keeps the AD graph shallow
    when summing many terms.
    """
    if len(children) == 1:
        return children[0]
    mid = len(children) // 2
    return _diffengine.make_add(chain_add(children[:mid]), chain_add(children[mid:]))


def make_sparse_left_matmul(param_node, child, A):
    if not isinstance(A, sparse.csr_array):
        A = sparse.csr_array(A)
    return _diffengine.make_left_matmul(
        param_node, child, 'sparse',
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int32, copy=False),
        A.indptr.astype(np.int32, copy=False),
        A.shape[0], A.shape[1])


def make_dense_left_matmul(param_node, child, A):
    m, n = A.shape
    return _diffengine.make_left_matmul(
        param_node, child, 'dense', A.flatten(order='C'), m, n)


def make_sparse_right_matmul(param_node, child, A):
    if not isinstance(A, sparse.csr_array):
        A = sparse.csr_array(A)
    return _diffengine.make_right_matmul(
        param_node, child, 'sparse',
        A.data.astype(np.float64, copy=False),
        A.indices.astype(np.int32, copy=False),
        A.indptr.astype(np.int32, copy=False),
        A.shape[0], A.shape[1])


def make_dense_right_matmul(param_node, child, A):
    m, n = A.shape
    return _diffengine.make_right_matmul(
        param_node, child, 'dense', A.flatten(order='C'), m, n)


def make_constant_quad_form(x_c, P_val, n):
    """Build the engine quad_form node for a plain-constant P, routed by
    content, not container: a sparse-stored or mostly-zero P takes the sparse
    CSR binding (a dense-stored diagonal like H * np.eye(n) would otherwise
    become a dense n^2 Hessian block), a genuinely dense P the dense binding.
    Explicitly stored zeros are dropped so the engine sees the true pattern.
    Shared by the DCP-path (SymbolicQuadForm) and NLP-path (QuadForm)
    converters so the routing policy cannot drift between them."""
    if not sparse.issparse(P_val):
        P_dense = to_dense_float(P_val)
        density = np.count_nonzero(P_dense) / P_dense.size if P_dense.size else 1.0
        if density >= s.SPARSE_DENSITY_THRESHOLD:
            return _diffengine.make_quad_form(
                None, x_c, "dense", P_dense.flatten(order='F'), n)
        P_val = sparse.csr_array(P_dense)
    P_csr = P_val.tocsr()
    if (P_csr.data == 0).any():
        # tocsr() on an already-CSR P aliases the caller's arrays; copy
        # before eliminate_zeros() so the user's matrix is not mutated.
        P_csr = P_csr.copy()
        P_csr.eliminate_zeros()
    return _diffengine.make_quad_form(
        None, x_c, "sparse",
        P_csr.data.astype(np.float64),
        P_csr.indices.astype(np.int32),
        P_csr.indptr.astype(np.int32),
        P_csr.shape[0], P_csr.shape[1])


def build_var_dict(inverse_data):
    """Build {var_id: C variable capsule} mapping from InverseData."""
    n_vars = inverse_data.x_length
    var_dict = {}
    for var_id, (offset, _) in inverse_data.id_map.items():
        d1, d2 = normalize_shape(inverse_data.var_shapes[var_id])
        var_dict[var_id] = _diffengine.make_variable(d1, d2, offset, n_vars)
    return var_dict, n_vars


def build_param_dict(parameters, inverse_data):
    """Build {param_id: C parameter capsule} mapping from InverseData.

    `parameters` is an iterable of `cvxpy.Parameter` (e.g. `problem.parameters()`).
    """
    n_vars = inverse_data.x_length
    params_by_id = {p.id: p for p in parameters}
    param_dict = {}
    for param_id, offset in inverse_data.param_id_map.items():
        # this is needed to not get key errors with Constants.
        if param_id not in inverse_data.param_shapes:
            continue
        d1, d2 = normalize_shape(inverse_data.param_shapes[param_id])
        p = to_dense_float(params_by_id[param_id].value)
        param_dict[param_id] = _diffengine.make_parameter(
            d1, d2, offset, n_vars, p.flatten(order='F'))
    return param_dict
