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

import inspect
import sys

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms import QuadForm
from cvxpy.expressions.constants import Constant


def attributes():
    """Return all attributes, i.e. all functions in this module except this function"""
    this_module_name = __name__
    return [obj for name, obj in inspect.getmembers(sys.modules[this_module_name])
            if (inspect.isfunction(obj) and
            name != 'attributes')]


def is_qp_objective(objective):
    expr = objective.expr
    if not type(expr) == AddExpression:
        return False
    if not len(expr.args) == 2:
        return False
    if not type(expr.args[0]) == QuadForm or not type(expr.args[1]) == MulExpression:
        return False
    if not expr.args[1].is_affine():
        return False
    return True


def is_cone_objective(objective):
    expr = objective.expr
    if not expr.is_affine():
        return False
    if not type(expr) == AddExpression:
        return False
    if not len(expr.args) == 2:
        return False
    if not type(expr.args[0]) == MulExpression or not type(expr.args[1]) == Constant:
        return False
    return True
