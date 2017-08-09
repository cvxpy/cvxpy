"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.expressions.expression import Expression


def diff(x, k=1, axis=0):
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
    elif axis == 0:
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
