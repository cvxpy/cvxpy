"""
Copyright 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
    placeholder = Variable(quad_form.shape,
                           var_id=quad_form.id)
    expr.args[idx] = placeholder
    quad_forms[placeholder.id] = (expr, idx, quad_form)
    return quad_forms


def restore_quad_forms(expr, quad_forms) -> None:
    for idx, arg in enumerate(expr.args):
        if isinstance(arg, Variable) and arg.id in quad_forms:
            expr.args[idx] = quad_forms[arg.id][2]
        else:
            restore_quad_forms(arg, quad_forms)
