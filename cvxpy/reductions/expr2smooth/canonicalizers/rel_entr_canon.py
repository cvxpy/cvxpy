"""
Copyright 2025 CVXPY developers

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

import numpy as np

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.elementwise.entr import entr
from cvxpy.atoms.elementwise.log import log
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.expr2smooth.canonicalizers.entr_canon import entr_canon
from cvxpy.reductions.expr2smooth.canonicalizers.log_canon import log_canon
from cvxpy.reductions.expr2smooth.canonicalizers.multiply_canon import multiply_canon


def rel_entr_canon(expr, args):

    # if the first argument is constant we canonicalize using log
    if args[0].is_constant():
        _log = log(args[1])
        log_expr, constr_log = log_canon(_log, _log.args)
        x = args[0].value
        return  x * np.log(x) - multiply(x, log_expr), constr_log

    # if the second argument is constant we canonicalize using entropy
    if args[1].is_constant():
        _entr = entr(args[0])
        entr_expr, constr_entr = entr_canon(_entr, _entr.args)
        _mult = multiply(args[0], np.log(args[1].value))
        mult_expr, constr_mult = multiply_canon(_mult, _mult.args)
        return -entr_expr - mult_expr, constr_entr + constr_mult

    # here we know that neither argument is constant
    t1 = Variable(args[0].shape, bounds=[0, None])
    t2 = Variable(args[1].shape, bounds=[0, None])
    constraints = [t1 == args[0], t2 == args[1]]

    if args[0].value is not None and np.min(args[0].value) >= 1:
        t1.value = args[0].value
    else:
        t1.value = expr.point_in_domain(argument=0)

    if args[1].value is not None and np.all(args[1].value >= 1):
        t2.value = args[1].value
    else:
        t2.value = expr.point_in_domain(argument=1)
    
    return expr.copy([t1, t2]), constraints
