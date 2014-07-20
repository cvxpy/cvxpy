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
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm2 import norm2
from cvxpy.atoms.norm_inf import normInf
from cvxpy.atoms.norm_nuc import normNuc
from cvxpy.expressions.variables import Variable
from cvxpy.atoms.affine.hstack import hstack

def norm_mixed(X, p=2, q=1):
    """L2,1 norm; :math:` (\sum_k (\sum_l \lvert x_{k,l} \rvert )^q/p)^{1/q}`.

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
    def norm_selector(order):
        if order == 1:
            return norm1
        elif order == 2:
            return norm2
        elif order == "inf":
            return normInf
        elif order == "nuc":
            return normNuc
        else:
            raise Exception("Invalid value %s for p." % p)

    pnorm = norm_selector(p)
    qnorm = norm_selector(q)
    
    vecnorms = [ pnorm(X[i, :]) for i in range(X.shape.cols) ]

    return qnorm(hstack(*vecnorms))

