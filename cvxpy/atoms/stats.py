"""
Copyright 2013 CVXPY Developers

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

import numbers

import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.atoms.affine.transpose import moveaxis
from cvxpy.atoms.norm import norm
from cvxpy.atoms.sum_squares import sum_squares
from cvxpy.expressions.expression import Expression


def _axis_size(x, axis=None) -> int:
    """Return the number of entries reduced by an axis argument."""
    if axis is None:
        return x.size
    axes = normalize_axis_tuple(axis, len(x.shape), "axis")
    return int(np.prod([x.shape[a] for a in axes]))


def mean(x, axis=None, keepdims=False) -> Expression:
    """
    Returns the mean of x.
    """
    return cvxpy_sum(x, axis=axis, keepdims=keepdims) / _axis_size(x, axis)


def std(x, axis=None, keepdims=False, ddof=0) -> Expression:
    """
    Returns the standard deviation of x.

    `ddof` is the quantity to use in the Bessel correction.
    """
    if axis is None:
        return norm((x - mean(x)).flatten(order='F'), 2) / np.sqrt(x.size - ddof)

    centered = x - mean(x, axis, True)
    scale = np.sqrt(_axis_size(x, axis) - ddof)
    if isinstance(axis, numbers.Integral):
        return norm(centered, 2, axis=axis, keepdims=keepdims) / scale

    # A tuple of axes pools the entries of every axis it names, which is one
    # vector per remaining position. norm takes a single axis, so the pooled
    # axes are moved to the front and flattened into one, in Fortran order,
    # and the result is folded back into the shape those axes left behind.
    axes = normalize_axis_tuple(axis, x.ndim, "axis")
    kept = tuple(d for d in range(x.ndim) if d not in axes)
    moved = moveaxis(centered, axes, range(len(axes)))
    pooled = norm(reshape(moved, (_axis_size(x, axis), -1), order='F'), 2, axis=0) / scale
    out_shape = tuple(x.shape[d] for d in kept)
    if keepdims:
        out_shape = tuple(1 if d in axes else x.shape[d] for d in range(x.ndim))
    return reshape(pooled, out_shape, order='F')


def var(x, axis=None, keepdims=False, ddof=0) -> Expression:
    """
    Returns the variance of x.

    `ddof` is the quantity to use in the Bessel correction.
    """
    return sum_squares(
        x - mean(x, axis, True),
        axis=axis,
        keepdims=keepdims,
    ) / (_axis_size(x, axis) - ddof)
