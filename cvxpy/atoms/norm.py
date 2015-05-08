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

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.atoms.sigma_max import sigma_max
from cvxpy.atoms.pnorm import pnorm


def norm(x, p=2):
    """Wrapper on the different norm atoms.

    Parameters
    ----------
    x : Expression or numeric constant
        The value to take the norm of.
    p : int or str, optional
        The type of norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    x = Expression.cast_to_const(x)
    if p == 1:
        return pnorm(x, 1)
    elif p == "inf":
        return pnorm(x, 'inf')
    elif p == "nuc":
        return normNuc(x)
    elif p == "fro":
        return pnorm(x, 2)
    elif p == 2:
        if x.is_matrix():
            return sigma_max(x)
        else:
            return pnorm(x, 2)
    else:
        return pnorm(x, p)
