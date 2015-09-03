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

from cvxpy.atoms.norm import norm
from cvxpy.atoms.quad_over_lin import quad_over_lin

def sum_squares(expr):
    """The sum of the squares of the entries.

    Parameters
    ----------
    expr: Expression
        The expression to take the sum of squares of.

    Returns
    -------
    Expression
        An expression representing the sum of squares.
    """
    return quad_over_lin(expr, 1)
