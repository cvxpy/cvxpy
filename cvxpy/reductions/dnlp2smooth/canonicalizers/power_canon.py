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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth.canonicalizers.div_canon import div_canon

MIN_INIT = 1e-3

def power_canon(expr, args):
    x = args[0]
    p = expr.p_used
    shape = expr.shape
    if p == 0:
        return Constant(np.ones(shape)), []
    elif p == 1:
        return x, []
    elif isinstance(p, int) and p > 1:
        return expr.copy(args), []
    elif p > 0:
        t = Variable(shape, nonneg=True)
        if x.value is not None:
            t.value = np.maximum(x.value, MIN_INIT)
        return expr.copy([t]), [t == x]
    else:
        # p < 0, so -p > 0. Canonicalize x^{-p} first, then wrap in 1/(...).
        inner_power_expr = x ** (-p)
        canon_inner, inner_constrs = power_canon(inner_power_expr, inner_power_expr.args)
        div_expr = Constant(np.ones(shape)) / canon_inner
        div_canon_expr, div_constr = div_canon(div_expr, div_expr.args)
        return div_canon_expr, inner_constrs + div_constr
