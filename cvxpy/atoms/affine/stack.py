"""NumPy-style stacking that inserts a new axis for CVXPY Expressions.

This mirrors the semantics of ``np.stack``, where they make sense, for symbolic
expressions: every input must share an identical shape, and the result gains a
new axis of length ``len(arrays)`` at the requested position. Options that are
specific to NumPy's ndarray type system (for example ``out`` or ``dtype``) are
intentionally unsupported.

Examples
--------
>>> import cvxpy as cp
>>> a = cp.Parameter((3,))
>>> b = cp.Parameter((3,))
>>> y = cp.stack([a, b], axis=0)
>>> y.shape
(2, 3)
>>> z = cp.stack([a, b], axis=-1)
>>> z.shape
(3, 2)
"""

from __future__ import annotations

from typing import Iterable, Sequence

from cvxpy.atoms.affine.concatenate import concatenate
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.expression import Expression


def _as_expression(obj) -> Expression:
    """Cast scalars/arrays to ``Constant`` while leaving Expressions intact."""
    return obj if isinstance(obj, Expression) else Expression.cast_to_const(obj)


def stack(arrays: Sequence[object] | Iterable[object], axis: int = 0) -> Expression:
    """Join a sequence of expressions along a new axis.

    Parameters
    ----------
    arrays
        Sequence of expressions (or array-likes) that all have the same shape.
    axis
        Index of the new axis in the result. Values in ``[-(ndim + 1), ndim``
        ``+ 1)`` are accepted, following ``numpy.stack``.

    Returns
    -------
    Expression
        Expression whose shape equals the common input shape with the new axis
        inserted at ``axis`` and length ``len(arrays)`` along that axis.

    Raises
    ------
    TypeError
        If ``axis`` is not an integer.
    ValueError
        If ``arrays`` is empty, shapes differ, or ``axis`` is out of bounds.
    """
    xs = [_as_expression(arg) for arg in arrays]
    if not xs:
        raise ValueError("need at least one array to stack")

    if not isinstance(axis, int):
        raise TypeError(f"axis must be an int; received {type(axis).__name__}")

    shapes = {expr.shape for expr in xs}
    if len(shapes) != 1:
        raise ValueError(
            "all input arrays must have the same shape; got "
            f"{sorted(shapes)}"
        )

    base_shape = xs[0].shape
    result_ndim = len(base_shape) + 1
    if not (-result_ndim <= axis < result_ndim):
        raise ValueError(
            f"axis {axis} is out of bounds for result ndim {result_ndim}"
        )

    axis_index = axis if axis >= 0 else axis + result_ndim
    # Slice the shape so we can splice a singleton axis where ``axis`` points.
    prefix = base_shape[:axis_index]
    suffix = base_shape[axis_index:]
    # Reshape each argument to inject the new length-1 axis before concatenating.
    reshaped = [
        reshape(expr, prefix + (1,) + suffix, order='F')
        for expr in xs
    ]

    return concatenate(reshaped, axis=axis_index)


__all__ = ["stack"]
