"""
Copyright, the CVXPY Ashok Viswanathan

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
from typing import Optional, Union

import numpy as np

from cvxpy.atoms.affine.sum import sum as cvx_sum
from cvxpy.expressions.expression import Expression

from .trapz import trapz


def _slice(expr: Expression, axis: int, idx: Union[int, slice]) -> Expression:
    """Helper to slice along a specific axis for CVXPY Expressions."""
    index = [slice(None)] * len(expr.shape)
    index[axis] = idx
    return expr[tuple(index)]


def simpson(
    y: Expression,
    x: Optional[np.ndarray] = None,
    dx: float = 1.0,
    axis: int = -1,
    even: str = "avg"
) -> Expression:
    """
    CVXPY-compatible version of scipy.integrate.simpson.

    Parameters:
    - y: CVXPY Expression to integrate.
    - x: Optional 1D array of sample points along `axis`.
    - dx: Spacing if `x` is not provided.
    - axis: Axis over which to integrate.
    - even: Strategy for even-length arrays. Options:
        'avg' (default): average of 'first' and 'last'
        'first': use first segment with trapz
        'last': use last segment with trapz

    Returns:
    - CVXPY Expression representing the integral.
    """
    axis = axis % len(y.shape)
    N = y.shape[axis]

    if N < 3:
        raise ValueError("Simpson's rule requires at least 3 points.")

    if N % 2 == 0:
        if even == "avg":
            first = simpson(_slice(y, axis, slice(0, N - 1)), x=x[:-1] if x is not None else None,
                             dx=dx, axis=axis)
            second = simpson(_slice(y, axis, slice(1, N)), x=x[1:] if x is not None else None,
                              dx=dx, axis=axis)
            return 0.5 * (first + second)
        elif even == "first":
            return (
                trapz(_slice(y, axis, slice(0, 2)), x=x[:2] if x is not None else None, 
                      dx=dx, axis=axis) +
                simpson(_slice(y, axis, slice(1, N)), x=x[1:] if x is not None else None, 
                        dx=dx, axis=axis)
            )
        elif even == "last":
            return (
                simpson(_slice(y, axis, slice(0, N - 1)), x=x[:-1] if x is not None else None, 
                        dx=dx, axis=axis) +
                trapz(_slice(y, axis, slice(N - 2, N)), x=x[-2:] if x is not None else None, 
                      dx=dx, axis=axis)
            )
        else:
            raise ValueError(f"Unknown even mode '{even}'")

    if x is not None:
        dx_vals = np.diff(x)
        if not np.allclose(dx_vals, dx_vals[0]):
            raise ValueError("x spacing must be uniform for Simpson's rule.")
        dx = dx_vals[0]

    return dx / 3 * (
        _slice(y, axis, 0) +
        _slice(y, axis, -1) +
        4 * cvx_sum(_slice(y, axis, slice(1, N - 1, 2))) +
        2 * cvx_sum(_slice(y, axis, slice(2, N - 1, 2)))
    )
