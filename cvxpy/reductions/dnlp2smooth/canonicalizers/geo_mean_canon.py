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
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.elementwise.log import log
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth.canonicalizers.log_canon import log_canon

MIN_INIT = 1e-4

def geo_mean_canon(expr, args):
    """
    Canonicalization for the geometric mean function.
    """
    t = Variable(expr.shape, nonneg=True)

    if args[0].value is not None and args[0].value > MIN_INIT:
        t.value = expr.numeric(args[0].value)
    else:
        t.value = np.ones(expr.shape)

    weights = np.array([float(w) for w in expr.w])
    log_expr = log(args[0])
    var, constr = log_canon(log_expr, expr.args)
    return t, [log(t) == sum(multiply(weights, var))] + constr
