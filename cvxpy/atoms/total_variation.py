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


def tv(value, *args):
    """Total variation of a vector, matrix, or list of matrices.

    Uses L1 norm of discrete gradients for vectors and
    L2 norm of discrete gradients for matrices.

    Parameters
    ----------
    value : Expression or numeric constant
        The value to take the total variation of.
    args : Matrix constants/expressions
        Additional matrices extending the third dimension of value.

    Returns
    -------
    Expression
        An Expression representing the total variation.
    """
    value = Expression.cast_to_const(value)
    if value.ndim == 0:
        raise ValueError("tv cannot take a scalar argument.")
    # L1 norm for vectors.
    elif value.ndim == 1:
        return norm(value[1:] - value[0:value.shape[0]-1], 1)
    # L2 norm for matrices.
    else:
        rows, cols = value.shape
        args = map(Expression.cast_to_const, args)
        values = [value] + list(args)
        diffs = []
        for mat in values:
            diffs += [
                mat[0:rows-1, 1:cols] - mat[0:rows-1, 0:cols-1],
                mat[1:rows, 0:cols-1] - mat[0:rows-1, 0:cols-1],
            ]
        length = diffs[0].shape[0]*diffs[1].shape[1]
        stacked = vstack([reshape(diff, (1, length), order='F') for diff in diffs])
        return sum(norm(stacked, p=2, axis=0))
