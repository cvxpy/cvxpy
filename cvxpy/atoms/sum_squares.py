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

from typing import Optional, Tuple, Union

from cvxpy.atoms.quad_over_lin import quad_over_lin


def sum_squares(
    expr,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False
):
    r"""The sum of the squares of the entries.

    .. math::

        f(x) = \sum_{ij} x_{ij}^2 = \|x\|_F^2

    Convex, always nonnegative, and quadratic.

    Parameters
    ----------
    expr : Expression
        The expression to take the sum of squares of.
    axis : int or tuple of int, optional
        The axis along which to compute the sum of squares (default: all elements).
    keepdims : bool, optional
        Whether to keep the reduced dimension (default: False).
    """
    return quad_over_lin(expr, 1, axis=axis, keepdims=keepdims)
