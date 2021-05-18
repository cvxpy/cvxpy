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

from cvxpy.expressions.expression import Expression


def diff(x, k: int = 1, axis: int = 0):
    """Vector of kth order differences.

    Takes in a vector of length n and returns a vector
    of length n-k of the kth order differences.

    diff(x) returns the vector of differences between
    adjacent elements in the vector, that is

    [x[2] - x[1], x[3] - x[2], ...]

    diff(x, 2) is the second-order differences vector,
    equivalently diff(diff(x))

    diff(x, 0) returns the vector x unchanged
    """
    x = Expression.cast_to_const(x)
    if (axis == 1 and x.ndim < 2) or x.ndim == 0:
        raise ValueError("Invalid axis given input dimensions.")
    elif axis == 1:
        x = x.T

    if k < 0 or k >= x.shape[axis]:
        raise ValueError("Must have k >= 0 and X must have < k elements along "
                         "axis")
    for i in range(k):
        if x.ndim == 2:
            x = x[1:, :] - x[:-1, :]
        else:
            x = x[1:] - x[:-1]
    return x.T if axis == 1 else x
