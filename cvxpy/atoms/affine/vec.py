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
from cvxpy.atoms.affine.reshape import reshape

def vec(X):
    """Flattens the matrix X into a vector in column-major order.

    Parameters
    ----------
    X : Expression or numeric constant
        The matrix to flatten.

    Returns
    -------
    Expression
        An Expression representing the flattened matrix.
    """
    X = Expression.cast_to_const(X)

    return reshape(X, X.size[0]*X.size[1], 1)
