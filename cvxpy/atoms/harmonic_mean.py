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
