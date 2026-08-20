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
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.norm import norm
from cvxpy.expressions.expression import Expression


def tv(value, *args) -> Expression:
    """Total variation of a vector, matrix, or list of matrices.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Channels are coupled: the gradients of every channel at a pixel are
    stacked into one vector before its L2 norm is taken, so this is colour
    total variation and not the sum of the per-channel ones. Differences are
    taken along the two spatial axes only.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of. A three-dimensional value
        of shape ``(m, n, k)`` is read as ``k`` channels of an ``m`` by ``n``
        image, equivalent to passing its slices as separate arguments.
    *args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast(value)
    if value.ndim == 0:
        raise ValueError("tv cannot take a scalar argument.")
    # L1 norm for vectors.
    elif value.ndim == 1:
        return norm(value[1:] - value[0:value.shape[0]-1], 1)

    extra = [Expression.cast(arg) for arg in args]
    # L2 norm for matrices.
    if value.ndim == 2:
        values = [value] + extra
    elif value.ndim == 3:
        # The trailing axis holds channels, which is what *args already
        # extends: an (m, n, k) array means the same as passing its k slices
        # separately. Differences stay along the two spatial axes, so this is
        # colour total variation rather than a variation over a volume.
        values = [value[:, :, i] for i in range(value.shape[2])] + extra
    else:
        raise ValueError("tv cannot have input arrays with more than 3 dimensions.")

    rows, cols = values[0].shape
    diffs = []
    for mat in values:
        diffs += [
            mat[0:rows-1, 1:cols] - mat[0:rows-1, 0:cols-1],
            mat[1:rows, 0:cols-1] - mat[0:rows-1, 0:cols-1],
        ]
    length = diffs[0].shape[0]*diffs[1].shape[1]
    stacked = vstack([reshape(diff, (1, length), order='F') for diff in diffs])
    return sum(norm(stacked, p=2, axis=0))
