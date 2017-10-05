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

from cvxpy.atoms import sum_squares
from cvxpy.atoms.quad_form import decomp_quad
from cvxpy.expressions.constants import Constant
from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_over_lin_canon import quad_over_lin_canon


def quad_form_canon(expr, args):
    scale, M1, M2 = decomp_quad(args[1].value)
    if M1.size > 0:
        expr = sum_squares(Constant(M1.T) * args[0])
    if M2.size > 0:
        scale = -scale
        expr = sum_squares(Constant(M2.T) * args[0])
    obj, constr = quad_over_lin_canon(expr, expr.args)
    return scale * obj, constr
