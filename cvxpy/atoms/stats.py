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

from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.atoms.norm import norm
from cvxpy.atoms.sum_squares import sum_squares


def _axis_size(x, axis=None) -> int:
    """Return the number of entries reduced by an axis argument."""
    if axis is None:
        return x.size
    axes = normalize_axis_tuple(axis, len(x.shape), "axis")
    return int(np.prod([x.shape[a] for a in axes]))


def mean(x, axis=None, keepdims=False):
    """
    Returns the mean of x.
    """
    return cvxpy_sum(x, axis=axis, keepdims=keepdims) / _axis_size(x, axis)


def std(x, axis=None, keepdims=False, ddof=0):
    """
    Returns the standard deviation of x.

    `ddof` is the quantity to use in the Bessel correction.
    """
    if axis is None:
        return norm((x - mean(x)).flatten(order='F'), 2) / np.sqrt(x.size - ddof)
    elif isinstance(axis, numbers.Integral):
        return norm(x - mean(x, axis, True), 2, axis=axis, keepdims=keepdims) \
                / np.sqrt(_axis_size(x, axis) - ddof)
    else:
        raise ValueError("cp.std doesn't yet support tuple axis values.")


def var(x, axis=None, keepdims=False, ddof=0):
    """
    Returns the variance of x.

    `ddof` is the quantity to use in the Bessel correction.
    """
    return sum_squares(
        x - mean(x, axis, True),
        axis=axis,
        keepdims=keepdims,
    ) / (_axis_size(x, axis) - ddof)
