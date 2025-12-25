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

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.variable import Variable


def cumsum_canon(expr, args):
    """Cumulative sum.
    """
    X = args[0]
    axis = expr.axis

    # Handle axis=None case: flatten in C order, then use axis=0 logic
    if axis is None:
        # Flatten X in C order
        total_size = X.size
        X_flat = reshape(X, (total_size,), order='C')

        # Now treat as 1D with axis=0
        Y = Variable((total_size,))

        # X_flat[1:] = Y[1:] - Y[:-1]
        # Y[0] = X_flat[0]
        constr = [X_flat[1:] == Y[1:] - Y[:-1],
                  Y[0:1] == X_flat[0:1]]
        return Y, constr

    ndim = len(expr.shape)

    # Normalize negative axis
    if axis < 0:
        axis = ndim + axis

    # If only one element along this axis, cumsum is identity
    if expr.shape[axis] == 1:
        return X, []

    # Implicit O(n) definition:
    # X = Y[1:,:] - Y[:-1, :] along the specified axis
    Y = Variable(expr.shape)

    # Build slices for "all but first" and "all but last" along axis
    # and for "first element" along axis
    slice_all = slice(None)
    slice_from_1 = slice(1, None)
    slice_to_minus1 = slice(None, -1)
    slice_first = slice(0, 1)

    # Create index tuples
    idx_from_1 = tuple(slice_from_1 if i == axis else slice_all for i in range(ndim))
    idx_to_minus1 = tuple(slice_to_minus1 if i == axis else slice_all for i in range(ndim))
    idx_first = tuple(slice_first if i == axis else slice_all for i in range(ndim))

    # X[from_1] = Y[from_1] - Y[to_minus1]
    # Y[first] = X[first]
    constr = [X[idx_from_1] == Y[idx_from_1] - Y[idx_to_minus1],
              Y[idx_first] == X[idx_first]]
    return Y, constr
