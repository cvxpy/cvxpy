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

import numpy as np

from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.atoms.norm import norm
from cvxpy.atoms.sum_squares import sum_squares


def mean(x, axis=None, keepdims=False):
    """
    Returns the mean of x.
    """
    if axis is None:
        return cvxpy_sum(x, axis, keepdims) / x.size
    elif axis in (0, 1):
        return cvxpy_sum(x, axis, keepdims) / x.shape[axis]
    else:
        raise ValueError("Invalid axis value.")


def std(x, axis=None, keepdims=False, ddof=0):
    """
    Returns the standard deviation of x.

    `ddof` is the quantity to use in the Bessel correction.
    """
    if axis is None:
        return norm((x - mean(x)).flatten(order='F'), 2) / np.sqrt(x.size - ddof)
    elif axis in (0, 1):
        return norm(x - mean(x, axis, True), 2, axis=axis, keepdims=keepdims) \
                / np.sqrt(x.shape[axis] - ddof)
    else:
        raise ValueError("Invalid axis value.")

def var(x, axis=None, keepdims=False, ddof=0):
    """
    Returns the variance of x.

    `ddof` is the quantity to use in the Bessel correction.
    """
    if axis is None:
        return sum_squares(x - mean(x)) / (x.size - ddof)
    elif axis in (0, 1):
        raise NotImplementedError(
            """axis and keepdims are not yet supported for var.
            Use square(std(...)) instead.
            """
        )
        # TODO when sum_squares implements axis and keepdims uncomment:
        # return sum_squares(x - mean(x, axis, True), 2, axis=axis, keepdims=keepdims) \
        #         / (x.shape[axis] - ddof)
    else:
        raise ValueError("Invalid axis value.")
