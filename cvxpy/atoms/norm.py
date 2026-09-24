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

import cvxpy
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.axis_atom import normalize_axis_tuple
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.pnorm import pnorm
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.expressions.expression import Expression


def norm(x, p: int | str = 2, axis=None, keepdims: bool = False) -> Expression:
    """Wrapper on the different norm atoms.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.  If `x` is 2D and `axis` is None,
        this function constructs a matrix norm.
    p : int or str, optional
        The type of norm. Valid options include any positive integer,
        'fro' (for frobenius), 'nuc' (sum of singular values), np.inf or
        'inf' (infinity norm).
    axis : None, int, or tuple of ints, optional
        The axis along which to apply the norm, if any. A tuple with a
        single element is equivalent to an integer axis. A tuple with two
        elements applies the corresponding matrix norm (p = 1, 2, 'fro',
        'nuc', or inf) to each slice spanned by the remaining axes, matching
        NumPy semantics. Tuples with more than two elements are not
        supported.
    keepdims: If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    x = Expression.cast(x)
    if isinstance(axis, tuple):
        if len(axis) == 0:
            raise ValueError("The axis parameter must not be an empty tuple.")
        elif len(axis) == 1:
            # A one-element tuple is equivalent to an integer axis.
            axis = axis[0]
        else:
            return _norm_over_matrix_axes(x, p, axis, keepdims)
    # matrix norms take precedence
    if axis is None and x.ndim == 2:
        return _matrix_norm(x, p)
    else:
        if axis is not None and str(p).lower() in ("fro", "nuc"):
            # Both are matrix norms, defined over a whole matrix rather than
            # along one of its axes. Without this the fro branch below vecs
            # the argument first and then hands the axis to a 1-D expression,
            # which silently returned the norm of the whole array for axis=0.
            raise ValueError(
                f"The axis parameter is not supported for the '{p}' norm, which is "
                "defined over a whole matrix. Use norm(x, 2, axis=...) for the "
                "2-norm along an axis."
            )
        if p == 1 or x.is_scalar():
            return norm1(x, axis=axis, keepdims=keepdims)
        elif str(p).lower() == "inf":
            return norm_inf(x, axis=axis, keepdims=keepdims)
        elif str(p).lower() == "fro":
            # TODO should not work for vectors.
            return pnorm(vec(x, order='F'), 2)
        elif isinstance(p, str):
            raise RuntimeError(f'Unsupported norm option {p} for non-matrix.')
        else:
            return pnorm(x, p, axis=axis, keepdims=keepdims)


def _matrix_norm(x: Expression, p: int | str) -> Expression:
    """The matrix norm p of a 2-D expression x."""
    num_nontrivial_idxs = sum([d > 1 for d in x.shape])
    if p == 1:  # matrix 1-norm
        return cvxpy.atoms.max(norm1(x, axis=0))
    # Frobenius norm
    elif p == 'fro' or (p == 2 and num_nontrivial_idxs == 1):
        return pnorm(vec(x, order='F'), 2)
    elif p == 2:  # matrix 2-norm is largest singular value
        return sigma_max(x)
    elif p == 'nuc':  # the nuclear norm (sum of singular values)
        return normNuc(x)
    elif p in [np.inf, "inf", "Inf"]:  # the matrix infinity-norm
        return cvxpy.atoms.max(norm1(x, axis=1))
    else:
        raise RuntimeError('Unsupported matrix norm.')


def _norm_over_matrix_axes(
    x: Expression,
    p: int | str,
    axis: tuple[int, ...],
    keepdims: bool,
) -> Expression:
    """Apply a matrix norm over the two-axis slices of an N-D expression.

    For x with shape (n0, ..., nk) and a two-element axis tuple, the matrix
    norm specified by p is applied to every slice obtained by fixing the
    remaining axes, mirroring np.linalg.norm semantics. The per-slice norms
    are stacked into the shape of the remaining axes (or the keepdims shape).
    """
    if len(set(axis)) != len(axis):
        raise ValueError("The axis parameter must not contain duplicate entries.")
    if len(axis) > 2:
        raise NotImplementedError(
            "norm() with more than two axis entries is not supported. "
            "Two-entry tuples are interpreted as matrix norms over the remaining "
            "slices, matching NumPy semantics."
        )
    axes = normalize_axis_tuple(axis, x.ndim)
    if len(axes) > 2:
        raise NotImplementedError(
            "norm() with more than two axis entries is not supported. "
            "Two-entry tuples are interpreted as matrix norms over the "
            "remaining slices, matching NumPy semantics."
        )
    remaining = tuple(a for a in range(x.ndim) if a not in axes)
    if len(remaining) > 2:
        raise NotImplementedError(
            "norm() with a two-element axis tuple is only supported for "
            "expressions with at most 4 dimensions."
        )
    if x.ndim == 2:
        # The tuple spans both axes of a matrix: NumPy returns a scalar
        # (or a (1, 1) array with keepdims=True).
        result = _matrix_norm(x, p)
        if keepdims:
            result = reshape(result, (1, 1), order='F')
        return result
    remaining_shape = tuple(x.shape[a] for a in remaining)
    if keepdims:
        target = tuple(
            1 if a in axes else x.shape[a] for a in range(x.ndim)
        )
    else:
        target = remaining_shape
    if x.size == 0:
        # NumPy defines every supported matrix norm of an empty array as 0,
        # so the result is a zero constant of the output shape. This covers
        # both an empty remaining axis (zero-size output) and empty reduced
        # axes (all slices empty).
        return cvxpy.Constant(np.zeros(target))
    # Compute the matrix norm of each slice; np.ndindex enumerates the
    # remaining coordinates in C order, matching NumPy result layout.
    entries = []
    for coords in np.ndindex(*remaining_shape):
        index = [slice(None)] * x.ndim
        for a, c in zip(remaining, coords):
            index[a] = c
        entries.append(_matrix_norm(x[tuple(index)], p))
    if len(remaining) == 0:
        return entries[0]
    # Stack the scalar entries: hstack the (1,)-shaped entries, then reshape
    # into the remaining shape (or the keepdims shape).
    flat = hstack([reshape(e, (1,), order='F') for e in entries])
    return reshape(flat, target, order='C')


def norm2(x, axis=None):
    """The 2-norm of x.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.  If `x` is 2D and `axis` is None,
        this function constructs a matrix norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    return norm(x, p=2, axis=axis)
