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
from cvxpy.atoms.norm import norm
from cvxpy.expressions.variables import Variable
from cvxpy.atoms.affine.hstack import hstack

def mixed_norm(X, p=2, q=1):
    """Lp,q norm; :math:` (\sum_k (\sum_l \lvert x_{k,l} \rvert )^q/p)^{1/q}`.

    Parameters
    ----------
    X : Expression or numeric constant
        The matrix to take the l_{p,q} norm of.
    p : int or str, optional
        The type of inner norm.
    q : int or str, optional
        The type of outer norm.

    Returns
    -------
    Expression
        An Expression representing the mixed norm.
    """
    X = Expression.cast_to_const(X)

    # inner norms
    vecnorms = [ norm(X[i, :], p) for i in range(X.size[0]) ]

    # outer norm
    return norm(hstack(*vecnorms), q)

