"""
Copyright 2025 CVXYPY Developers

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
from __future__ import annotations

import numbers
from typing import Tuple

from cvxpy.atoms.affine.reshape import reshape


def squeeze(expr, axis: int | Tuple[int, ...] | None = None):
    """
    Squeeze the expression.

    This operation removes dimensions of size 1
    from the shape of the input expression along
    the specified axes.

    Parameters
    ----------
    expr : Expression
       The expression to squeeze
    axis :
        The axis or axes along which to squeeze.
    """
    shape = _get_squeezed_shape(expr.shape, axis)
    return reshape(expr, shape, order='F')


def _get_squeezed_shape(
    shape: Tuple[int, ...],
    axis: int | Tuple[int, ...] | None,
) -> Tuple[int, ...]:
    if axis is None:
        axis = tuple(i for i, d in enumerate(shape) if d == 1)
    else:
        if isinstance(axis, numbers.Integral):
            axis = (int(axis),)
        axis = tuple(int(a) + (len(shape) if a < 0 else 0) for a in axis)

    for a in axis:
        if a < 0 or a >= len(shape):
            msg = f"Axis {a} is out of bounds for array of dimension {len(shape)}."
            raise ValueError(msg)
        if shape[a] != 1:
            msg = f"Cannot squeeze axis {a} with size {shape[a]}."
            raise ValueError(msg)
    return tuple(d for i, d in enumerate(shape) if i not in axis)
