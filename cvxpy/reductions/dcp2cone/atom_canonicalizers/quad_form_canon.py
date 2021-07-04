"""
Copyright 2013 Steven Diamond

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

from cvxpy.atoms import sum_squares
from cvxpy.atoms.quad_form import decomp_quad
from cvxpy.expressions.constants import Constant
from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_over_lin_canon import (
    quad_over_lin_canon,)


def quad_form_canon(expr, args):
    # TODO this doesn't work with parameters!
    scale, M1, M2 = decomp_quad(args[1].value)
    # Special case where P == 0.
    if M1.size == M2.size == 0:
        return Constant(0), []

    if M1.size > 0:
        expr = sum_squares(Constant(M1.T) @ args[0])
    if M2.size > 0:
        scale = -scale
        expr = sum_squares(Constant(M2.T) @ args[0])
    obj, constr = quad_over_lin_canon(expr, expr.args)
    return scale * obj, constr
