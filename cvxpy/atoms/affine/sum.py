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

from affine_atom import AffAtom
from ... import interface as intf

def sum(X):
    """Returns an expression representing the sum of the expression's entries.

    Parameters
    ----------
    X : Expression
        An Expression.

    Returns
    -------
    Expession
        A scalar Expression.
    """
    X = AffAtom.cast_to_const(X)
    rows, cols = X.size
    lh_ones = intf.DEFAULT_INTERFACE.ones(1, rows)
    rh_ones = intf.DEFAULT_INTERFACE.ones(cols, 1)
    return lh_ones*X*rh_ones
