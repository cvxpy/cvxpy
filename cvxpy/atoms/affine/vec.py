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

import warnings
from typing import Literal

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.expression import DEFAULT_ORDER_DEPRECATION_MSG, Expression


def vec(X, order: Literal["F", "C", None] = None):
    """Flattens the matrix X into a vector.

    Parameters
    ----------
    X : Expression or numeric constant
        The matrix to flatten.
    order: column-major ('F') or row-major ('C') order.

    Returns
    -------
    Expression
        An Expression representing the flattened matrix.
    """
    if order is None:
        vec_order_warning = DEFAULT_ORDER_DEPRECATION_MSG.replace("FUNC_NAME", "vec")
        warnings.warn(vec_order_warning, FutureWarning)
        order = 'F'
    assert order in ['F', 'C']
    X = Expression.cast_to_const(X)
    return reshape(X, (X.size,), order)
