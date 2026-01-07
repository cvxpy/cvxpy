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

from numpy.lib.array_utils import normalize_axis_index

from cvxpy.expressions.expression import Expression


def diff(x, k: int = 1, axis: int = 0):
    """Computes kth order differences along the specified axis.

    Takes in an array and returns an array with the kth order differences
    along the given axis. The output shape is the same as the input except
    the size along the specified axis is reduced by k.

    diff(x) returns the differences between adjacent elements along axis 0:
        [x[1] - x[0], x[2] - x[1], ...]

    diff(x, 2) is the second-order differences, equivalently diff(diff(x))

    diff(x, 0) returns the array x unchanged

    Parameters
    ----------
    x : Expression or array-like
        Input array.
    k : int, optional
        The number of times values are differenced. Default is 1.
    axis : int, optional
        The axis along which the difference is taken. Default is 0.
        Note: NumPy's np.diff uses axis=-1 as default.

    Returns
    -------
    Expression
        The kth order differences along the specified axis.
    """
    x = Expression.cast_to_const(x)

    # Validate and normalize axis (handles negative indices)
    if x.ndim == 0:
        raise ValueError("Invalid axis given input dimensions.")
    axis = normalize_axis_index(axis, x.ndim)

    # Validate k
    if k < 0 or k >= x.shape[axis]:
        raise ValueError("Must have k >= 0 and X must have < k elements along "
                         "axis")

    # Apply k iterations of first-order difference along axis
    for _ in range(k):
        slices_upper = [slice(None)] * x.ndim
        slices_upper[axis] = slice(1, None)

        slices_lower = [slice(None)] * x.ndim
        slices_lower[axis] = slice(None, -1)

        x = x[tuple(slices_upper)] - x[tuple(slices_lower)]

    return x
