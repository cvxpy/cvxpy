"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import Maximize, Minimize


def canonicalize_tree(expr, canon_methods):
    canon_args = []
    constrs = []
    for arg in expr.args:
        canon_arg, c = canonicalize_tree(arg, canon_methods)
        canon_args += [canon_arg]
        constrs += c
    canon_expr, c = canonicalize_expr(expr, canon_args, canon_methods)
    constrs += c
    return canon_expr, constrs


def canonicalize_expr(expr, args, canon_methods):
    if isinstance(expr, Minimize):
        return Minimize(*args), []
    elif isinstance(expr, Maximize):
        return Maximize(*args), []
    elif isinstance(expr, Variable):
        return expr, []
    elif isinstance(expr, Constant):
        return expr, []
    elif isinstance(expr, Constraint):
        return type(expr)(*args), []
    elif expr.is_atom_convex() and expr.is_atom_concave():
        if isinstance(expr, AddExpression):
            expr = type(expr)(args)
        else:
            expr = type(expr)(*args)
        return expr, []
    else:
        return canon_methods[type(expr)](expr, args)
