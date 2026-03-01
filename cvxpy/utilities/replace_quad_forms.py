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

from cvxpy.atoms.quad_form import QuadForm, SymbolicQuadForm
from cvxpy.expressions.variable import Variable
from cvxpy.lin_ops.lin_op import LinOp


def replace_quad_forms(expr, quad_forms):
    """Replace QuadForm/SymbolicQuadForm nodes with dummy Variables.

    Returns a new expression tree (the original is not modified) and a dict
    mapping each dummy variable's id to the original QuadForm atom.

    Parameters
    ----------
    expr : LinOp or Expression
        The root of the expression tree to process.
    quad_forms : dict
        Accumulator mapping dummy_id -> quad_form_atom.

    Returns
    -------
    new_expr : LinOp or Expression
        A copy of `expr` with QuadForm nodes replaced by dummy Variables.
        Subtrees that contain no QuadForm nodes are shared (not copied).
    quad_forms : dict
        Updated mapping of dummy_id -> quad_form_atom.
    """
    new_args = list(expr.args)
    changed = False
    for idx, arg in enumerate(new_args):
        if isinstance(arg, (SymbolicQuadForm, QuadForm)):
            placeholder = Variable(arg.shape, var_id=arg.id)
            quad_forms[placeholder.id] = arg
            new_args[idx] = placeholder
            changed = True
        else:
            new_arg, quad_forms = replace_quad_forms(arg, quad_forms)
            if new_arg is not arg:
                new_args[idx] = new_arg
                changed = True

    if not changed:
        return expr, quad_forms

    if isinstance(expr, LinOp):
        new_expr = LinOp(expr.type, expr.shape, new_args, expr.data)
    else:
        new_expr = expr.copy(new_args)
    return new_expr, quad_forms
