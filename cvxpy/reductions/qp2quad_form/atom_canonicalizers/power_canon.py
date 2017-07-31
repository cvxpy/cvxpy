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

import numpy as np

from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable


def power_canon(expr, args):
    affine_expr = args[0]
    p = expr.p
    if expr.is_constant():
        return Constant(expr.value), []
    elif p == 0:
        return np.ones(affine_expr.shape), []
    elif p == 1:
        return affine_expr, []
    elif p == 2:
        t = Variable(affine_expr.shape)
        return SymbolicQuadForm(t, np.eye(t.size), expr), [affine_expr == t]
    raise ValueError("non-constant quadratic forms can't be raised to a power "
                     "greater than 2.")
