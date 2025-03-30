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
from typing import Optional

import numpy as np

from cvxpy.atoms.affine.binary_operators import MulExpression as cvx_multiply
from cvxpy.atoms.affine.sum import sum as cvx_sum
from cvxpy.expressions.expression import Expression


def trapz(
    y: Expression,
    x: Optional[np.ndarray] = None,
    dx: float = 1.0,
    axis: int = -1
) -> Expression:
    y_ndim = len(y.shape)
    axis = axis % y_ndim
    n = y.shape[axis]

    def slicer(start, stop):
        return tuple(
            slice(None) if i != axis else slice(start, stop)
            for i in range(y_ndim)
        )

    left = y[slicer(0, n - 1)]
    right = y[slicer(1, n)]

    if x is not None:
        dxs = np.diff(x)
        dxs_shape = [1] * y_ndim
        dxs_shape[axis] = len(dxs)
        dxs_broadcast = dxs.reshape(dxs_shape)
        return 0.5 * cvx_sum(cvx_multiply(left + right, dxs_broadcast))
    else:
        edge = 0.5 * (y[slicer(0, 1)] + y[slicer(n - 1, n)])
        center = cvx_sum(y[slicer(1, n - 1)])
        return dx * (edge + center)

# def trapz(
#     y: Expression,
#     x: Optional[np.ndarray] = None,
#     dx: float = 1.0,
#     axis: int = -1
# ) -> Expression:
#     y_ndim = len(y.shape)
#     axis = axis % y_ndim
#     n = y.shape[axis]

#     slicer = lambda start, stop: tuple(
#         slice(None) if i != axis else slice(start, stop)
#         for i in range(y_ndim)
#     )
#     left = y[slicer(0, n - 1)]
#     right = y[slicer(1, n)]

#     if x is not None:
#         dxs = np.diff(x)
#         dxs_shape = [1] * y_ndim
#         dxs_shape[axis] = len(dxs)
#         dxs_broadcast = dxs.reshape(dxs_shape)
#         return 0.5 * cvx_sum(cvx_multiply(left + right, dxs_broadcast))
#     else:
#         edge = 0.5 * (y[slicer(0,1)] + y[slicer(n-1, n)])
#         center = cvx_sum(y[slicer(1, n - 1)])
#         return dx * (edge + center)
