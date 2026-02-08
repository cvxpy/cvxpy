"""
Copyright 2013 Steven Diamond

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

from cvxpy.atoms import trace
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.lambda_max_canon import (
    lambda_max_canon,
)
from cvxpy.utilities.solver_context import SolverInfo


def _batched_trace(Z, batch_shape, n):
    """Compute trace of each (n, n) slice in a (*batch, n, n) expression.

    Returns an expression with shape batch_shape.
    """
    batch_size = int(np.prod(batch_shape))
    # Flatten Z to (batch_size * n * n, 1) in F-order,
    # then multiply by a coefficient matrix that picks diagonals.
    # In F-order layout of (batch_size, n, n), the diagonal entries of
    # the k-th matrix are at positions k + batch_size * (i * n + i)
    # for i = 0..n-1.
    flat = reshape(Z, (batch_size * n * n, 1), order='F')
    diag_indices = np.arange(n) * n + np.arange(n)  # i * n + i for F-order n x n
    # For each batch element b, the entry is at b + batch_size * diag_index
    import scipy.sparse as sp
    row_idx = np.repeat(np.arange(batch_size), n)
    col_idx = (np.tile(diag_indices, batch_size) * batch_size
               + np.repeat(np.arange(batch_size), n))
    coeff = Constant(sp.csc_array(
        (np.ones(batch_size * n), (row_idx, col_idx)),
        shape=(batch_size, batch_size * n * n)))
    result = coeff @ flat
    return reshape(result, batch_shape, order='F')


def lambda_sum_largest_canon(expr, args, solver_context: SolverInfo | None = None):
    """
    S_k(X) denotes lambda_sum_largest(X, k)
    t >= k S_k(X - Z) + trace(Z), Z is PSD
    implies
    t >= ks + trace(Z)
    Z is PSD
    sI >= X - Z (PSD sense)
    which implies
    t >= ks + trace(Z) >= S_k(sI + Z) >= S_k(X)
    We use the fact that
    S_k(X) = sup_{sets of k orthonormal vectors u_i}sum_{i}u_i^T X u_i
    and if Z >= X in PSD sense then
    sum_{i}u_i^T Z u_i >= sum_{i}u_i^T X u_i

    We have equality when s = lambda_k and Z diagonal
    with Z_{ii} = (lambda_i - lambda_k)_+
    """
    X = expr.args[0]
    k = expr.k
    n = X.shape[-1]
    if X.ndim == 2:
        Z = Variable((n, n), PSD=True)
        obj, constr = lambda_max_canon(expr, [X - Z])
        obj = k * obj + trace(Z)
        return obj, constr
    else:
        # nd case: X has shape (*batch, n, n)
        batch_shape = X.shape[:-2]
        Z = Variable(batch_shape + (n, n), PSD=True)

        obj, constr = lambda_max_canon(expr, [X - Z])

        # Batched trace via coefficient matrix
        batch_trace = _batched_trace(Z, batch_shape, n)
        obj = k * obj + batch_trace
        return obj, constr
