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

from cvxpy.atoms.lambda_sum_largest import lambda_sum_largest
from cvxpy.expressions.expression import Expression


def lambda_sum_smallest(X, k):
    r"""The sum of the smallest :math:`k` eigenvalues of a Hermitian matrix.

    .. math::

        f(X) = \sum_{i=n-k+1}^n \lambda_i(X)

    where :math:`\lambda_i(X)` is the :math:`i`-th largest eigenvalue of :math:`X`.

    Concave and always real-valued.

    Parameters
    ----------
    X : Expression
        The Hermitian matrix expression.
    k : int
        The number of eigenvalues to sum.
    """
    X = Expression.cast_to_const(X)
    return -lambda_sum_largest(-X, k)
