"""
Copyright 2017 Robin Verschueren

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

from cvxpy.expressions.variable import Variable
from cvxpy.atoms.quad_form import SymbolicQuadForm
from numpy import eye


def quad_over_lin_canon(expr, args):
    affine_expr = args[0]
    y = args[1]
    t = Variable(affine_expr.shape)
    return SymbolicQuadForm(t, eye(affine_expr.size)/y, expr), [affine_expr == t]
