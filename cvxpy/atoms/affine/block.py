"""
Copyright, the CVXPY authors

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

from cvxpy.atoms.affine.concatenate import concatenate
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.expression import Expression


def block(arrays) -> Expression:
    """Assemble an expression from nested lists of expressions.

    The innermost lists are concatenated along the last axis,
    the next level along the second-to-last axis, and so on.
    Lower-dimensional inputs are prepended with dimensions of size one.

    Parameters
    ----------
    arrays
        An expression or nested list of expressions.

    Returns
    -------
    Expression
        The assembled expression.
    """
    if not isinstance(arrays, list):
        return Expression.cast(arrays)

    list_depth = _list_depth(arrays)
    result_ndim = max(list_depth, _max_ndim(arrays))

    return _block(arrays, list_depth, result_ndim)


def _list_depth(arrays) -> int:
    if not isinstance(arrays, list):
        return 0

    if not arrays:
        raise ValueError("List at arrays cannot be empty")

    depths = [_list_depth(item) for item in arrays]

    if len(set(depths)) != 1:
        raise ValueError("List depths are mismatched")

    return 1 + depths[0]


def _max_ndim(arrays) -> int:
    if isinstance(arrays, list):
        return max(_max_ndim(item) for item in arrays)

    return Expression.cast(arrays).ndim


def _block(arrays, list_depth, result_ndim):
    if not isinstance(arrays, list):
        expr = Expression.cast(arrays)
        return _promote(expr, result_ndim)

    children = [
        _block(item, list_depth - 1, result_ndim)
        for item in arrays
    ]

    axis = result_ndim - list_depth
    return concatenate(children, axis=axis)


def _promote(expr: Expression, ndim: int) -> Expression:
    if expr.ndim == ndim:
        return expr

    new_shape = (1,) * (ndim - expr.ndim) + expr.shape
    return reshape(expr, new_shape, order="F")
