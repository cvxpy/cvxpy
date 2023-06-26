"""
Copyright 2022, adapted from Convex.jl.

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


def _term(expr, j: int, dims: Tuple[int], axis: Optional[int] = 0):
    """Helper function for partial trace.

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The 2D expression to take the partial trace of.
    j : int
        Term in the partial trace sum.
    dims : tuple of ints.
        A tuple of integers encoding the dimensions of each subsystem.
    axis : int
        The index of the subsystem to be traced out
        from the tensor product that defines expr.
    """
    # (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I) for all j's
    # in the system we want to trace out.
    # This function returns the jth term in the sum, namely
    # (I ⊗ <j| ⊗ I) x (I ⊗ |j> ⊗ I).
    a = sp.coo_matrix(([1.0], ([0], [0])))
    b = sp.coo_matrix(([1.0], ([0], [0])))
    for (i_axis, dim) in enumerate(dims):
        if i_axis == axis:
            v = sp.coo_matrix(([1], ([j], [0])), shape=(dim, 1))
            a = sp.kron(a, v.T)
            b = sp.kron(b, v)
        else:
            eye_mat = sp.eye(dim)
            a = sp.kron(a, eye_mat)
            b = sp.kron(b, eye_mat)
    return a @ expr @ b


# ruff: noqa: E501
def partial_trace(expr, dims: Tuple[int], axis: Optional[int] = 0):
    """
    Assumes :math:`\\texttt{expr} = X_1 \\otimes \\cdots \\otimes X_n` is a 2D Kronecker
    product composed of :math:`n = \\texttt{len(dims)}` implicit subsystems.
    Letting :math:`k = \\texttt{axis}`, the returned expression represents
    the *partial trace* of :math:`\\texttt{expr}` along its :math:`k^{\\text{th}}` implicit subsystem:

    .. math::

        \\text{tr}(X_k) (X_1 \\otimes \\cdots \\otimes X_{k-1} \\otimes X_{k+1} \\otimes \\cdots \\otimes X_n).

    Parameters
    ----------
    expr : :class:`~cvxpy.expressions.expression.Expression`
        The 2D expression to take the partial trace of.
    dims : tuple of ints.
        A tuple of integers encoding the dimensions of each subsystem.
    axis : int
        The index of the subsystem to be traced out
        from the tensor product that defines expr.
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
    return sum([_term(expr, j, dims, axis) for j in range(dims[axis])])
