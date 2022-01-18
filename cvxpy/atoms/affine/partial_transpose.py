"""
Copyright 2022, @duguyipiao.

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
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.atom import Atom


def _term(expr, i: int, j: int, dims: Tuple[int], axis: Optional[int] = 0):
    """Helper function for partial transpose.

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The expression to take the partial trace of.
    i : int
        Term in the partial trace sum.
    j : int
        Term in the partial trace sum.
    dims : tuple of ints.
        Whether to drop dimensions after summing.
    axis : int
        The axis along which to take the partial trace.
    """
    # (I ⊗ |i><j| ⊗ I) x (I ⊗ |i><j| ⊗ I) for all (i,j)'s
    # in the system we want to transpose.
    # This function returns the (i,j)-th term in the sum, namely
    # (I ⊗ |i><j| ⊗ I) x (I ⊗ |i><j| ⊗ I).
    a = sp.coo_matrix(([1.0], ([0], [0])))
    for (i_axis, dim) in enumerate(dims):
        if i_axis == axis:
            v = sp.coo_matrix(([1], ([i], [j])), shape=(dim, dim))
            a = sp.kron(a, v)
        else:
            eye_mat = sp.eye(dim)
            a = sp.kron(a, eye_mat)
    return a @ expr @ a


def partial_transpose(expr, dims: Tuple[int], axis: Optional[int] = 0):
    """Partial transpose of a matrix.

    Assumes expr = X1 \\odots ... \\odots Xn.
    Let axis=k be the dimension along which the partial transpose is taken.
    Returns X1 \\odots ... \\odots Xk^T \\odots ... \\odots Xn.

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The expression to take the partial trace of.
    dims : tuple of ints.
        Whether to drop dimensions after summing.
    axis : int
        The axis along which to take the partial trace.
    """
    expr = Atom.cast_to_const(expr)
    if expr.ndim < 2 or expr.shape[0] != expr.shape[1]:
        raise ValueError("Only supports square matrices.")
    if axis < 0 or axis >= len(dims):
        raise ValueError(
            f"Invalid axis argument, should be between 0 and {len(dims)}, got {axis}."
        )
    if expr.shape[0] != np.prod(dims):
        raise ValueError("Dimension of system doesn't correspond to dimension of subsystems.")
    return sum([
        _term(expr, i, j, dims, axis) for i in range(dims[axis]) for j in range(dims[axis])
    ])
