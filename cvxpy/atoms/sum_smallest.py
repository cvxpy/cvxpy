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

from cvxpy.atoms.sum_largest import sum_largest
from cvxpy.expressions.expression import Expression


def sum_smallest(x, k):
    r"""Sum of the smallest :math:`k` values.

    .. math::

        f(x) = \sum_{i=n-k+1}^n x_{[i]}

    where :math:`x_{[i]}` is the :math:`i`-th largest value of :math:`x`.

    Concave and always non-decreasing.

    Parameters
    ----------
    x : Expression
        the expression to take the sum of smallest values of.
    k : int
        the number of smallest values to sum.
    """
    x = Expression.cast_to_const(x)
    return -sum_largest(-x, k)
