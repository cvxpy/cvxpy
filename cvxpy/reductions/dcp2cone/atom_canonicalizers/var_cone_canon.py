"""
Copyright 2013 Steven Diamond, Eric Chu

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

from cvxpy.atoms.affine.reshape import reshape
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable, upper_tri_to_full


def var_cone_canon(expr, args):
    """Expand implicit constraints on variable.
    """
    # Convert attributes into constraints.
    new_attr = expr.attributes.copy()
    for key in ['nonneg', 'nonpos', 'symmetric', 'PSD', 'NSD']:
        if new_attr[key]:
            new_attr[key] = False

    if expr.is_symmetric():
        n = expr.shape[0]
        shape = (n*(n+1)//2, 1)
        upper_tri = Variable(shape, var_id=expr.id, **new_attr)
        fill_coeff = Constant(upper_tri_to_full(n))
        full_mat = fill_coeff*upper_tri
        obj = reshape(full_mat, (n, n))
    else:
        obj = Variable(expr.shape, var_id=expr.id, **new_attr)

    constr = []
    if expr.is_nonneg():
        constr.append(obj >= 0)
    elif expr.is_nonpos():
        constr.append(obj <= 0)
    elif expr.attributes['PSD']:
        constr.append(obj >> 0)
    elif expr.attributes['NSD']:
        constr.append(obj << 0)
    return (obj, constr)
