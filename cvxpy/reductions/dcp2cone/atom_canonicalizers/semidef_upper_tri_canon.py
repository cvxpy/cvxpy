
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
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.variables.symmetric import upper_tri_to_full


def semidef_upper_tri_canon(expr, args):
    """ The upper triangular part of a positive semidefinite variable. """
    upper_tri = Variable(expr.shape[0], 1, var_id=expr.id)
    fill_coeff = Constant(upper_tri_to_full(expr.n))
    full_mat = fill_coeff*upper_tri
    full_mat = reshape(full_mat, (expr.n, expr.n))
    return (upper_tri, [full_mat >> 0])
