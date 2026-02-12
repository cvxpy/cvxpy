"""
NumPy-compatible block implementation for CVXPY.
"""

from typing import Any

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.concatenate import concatenate


def _as_expr(x):
    return AffAtom.cast_to_const(x)


def _ndim(x):
    return _as_expr(x).ndim


def _max_ndim(arr):
    if isinstance(arr, list):
        return max(_max_ndim(a) for a in arr)
    return _ndim(arr)


def _block_depth(arr):
    depth = 0
    while isinstance(arr, list):
        depth += 1
        arr = arr[0]
    return depth


def _block_rec(arr, level, depth, ndim):
    if not isinstance(arr, list):
        return _as_expr(arr)

    sub = [_block_rec(a, level + 1, depth, ndim) for a in arr]

    axis = ndim - depth + level
    return concatenate(sub, axis=axis)


def block(arr: Any):
    if not isinstance(arr, list):
        raise ValueError("Input must be a nested list structure.")

    depth = _block_depth(arr)
    ndim = max(_max_ndim(arr), depth)

    def promote(x):
        if isinstance(x, list):
            return [promote(a) for a in x]
        expr = _as_expr(x)
        if expr.ndim < ndim:
            new_shape = (1,) * (ndim - expr.ndim) + expr.shape
            return expr.reshape(new_shape, order="F")
        return expr

    promoted = promote(arr)

    return _block_rec(promoted, level=0, depth=depth, ndim=ndim)
