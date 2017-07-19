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

from cvxpy.atoms.quad_form import SymbolicQuadForm, QuadForm
from cvxpy.expressions.variable import Variable


def replace_quad_forms(expr, quad_forms):
    for idx, arg in enumerate(expr.args):
        if isinstance(arg, SymbolicQuadForm) or isinstance(arg, QuadForm):
            quad_forms = replace_quad_form(expr, idx, quad_forms)
        else:
            quad_forms = replace_quad_forms(arg, quad_forms)
    return quad_forms


def replace_quad_form(expr, idx, quad_forms):
    quad_form = expr.args[idx]
    placeholder = Variable(quad_form.shape)
    expr.args[idx] = placeholder
    quad_forms[placeholder.id] = (expr, idx, quad_form)
    return quad_forms
