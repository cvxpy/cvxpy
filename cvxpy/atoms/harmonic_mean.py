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

from cvxpy.atoms.pnorm import pnorm
from cvxpy.expressions.expression import Expression


def harmonic_mean(x):
    """The harmonic mean of ``x``.

    Parameters
    ----------
    x : Expression or numeric
        The expression whose harmonic mean is to be computed. Must have
        positive entries.

    Returns
    -------
    Expression
        .. math::
            \\frac{n}{\\left(\\sum_{i=1}^{n} x_i^{-1} \\right)},

        where :math:`n` is the length of :math:`x`.
    """
    x = Expression.cast_to_const(x)
    # TODO(akshayka): Behavior of the below is incorrect when x has negative
    # entries. Either fail fast or provide a correct expression with
    # unknown curvature.
    return x.size*pnorm(x, -1)
