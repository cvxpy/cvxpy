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
    r"""The mean of all entries in an expression.

    .. math::

        f(x) = \frac{1}{n} \sum_{i,j} x_{ij}

    Affine and increasing.

    Parameters
    ----------
    x : Expression
        The expression to take the mean of.
    axis : int, optional
        The axis along which to apply the reduction (default: all elements).
    keepdims : bool, optional
        Whether to keep the reduced dimension (default: False).
    """
    if axis is None:
        return cvxpy_sum(x, axis, keepdims) / x.size
    elif axis in (0, 1):
        return cvxpy_sum(x, axis, keepdims) / x.shape[axis]
    else:
        raise UserWarning("cp.mean doesn't yet support axis values other than 0 or 1.")


def std(x, axis=None, keepdims=False, ddof=0):
    r"""The standard deviation of all entries in an expression.

    .. math::

        f(x) = \sqrt{\frac{1}{n-d} \sum_{ij} (x_{ij} - \bar{x})^2}

    where :math:`d` is the degrees of freedom (``ddof``).
    Convex and nonnegative.

    Parameters
    ----------
    x : Expression
        The expression to take the standard deviation of.
    axis : int, optional
        The axis along which to apply the reduction (default: all elements).
    keepdims : bool, optional
        Whether to keep the reduced dimension (default: False).
    ddof : int, optional
        Delta degrees of freedom (default: 0).
    """
    if axis is None:
        return norm((x - mean(x)).flatten(order='F'), 2) / np.sqrt(x.size - ddof)
    elif axis in (0, 1):
        return norm(x - mean(x, axis, True), 2, axis=axis, keepdims=keepdims) \
                / np.sqrt(x.shape[axis] - ddof)
    else:
        raise ValueError("cp.std doesn't yet support axis values other than 0 or 1.")

def var(x, axis=None, keepdims=False, ddof=0):
    r"""The variance of all entries in an expression.

    .. math::

        f(x) = \frac{1}{n-d} \sum_{ij} (x_{ij} - \bar{x})^2

    where :math:`d` is the degrees of freedom (``ddof``).
    Convex and nonnegative.

    Parameters
    ----------
    x : Expression
        The expression to take the variance of.
    axis : int, optional
        The axis along which to apply the reduction (default: all elements).
    keepdims : bool, optional
        Whether to keep the reduced dimension (default: False).
    ddof : int, optional
        Delta degrees of freedom (default: 0).
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
        raise ValueError("cp.var doesn't yet support axis values other than 0 or 1.")
