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

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.bounds import get_expr_bounds


# t = log_sum_exp(x) is equivalent exp(t) = sum(exp(x)), which is
# equivalent to sum(exp(x - t)) = 1. Now we introduce v = x - t,
# which must be nonpositive.
def log_sum_exp_canon(expr, args):
    x = args[0]
    bounds_t = get_expr_bounds(expr)
    t = Variable(expr.shape, bounds=bounds_t)

    # v = x - t, v <= 0
    x_bounds = get_expr_bounds(x)
    if x_bounds is not None and bounds_t is not None:
        v_lb = x_bounds[0] - bounds_t[1]
        v_ub = np.minimum(x_bounds[1] - bounds_t[0], 0)
        bounds_v = [v_lb, v_ub]
    elif x_bounds is not None:
        bounds_v = [None, np.minimum(x_bounds[1], 0)]
    else:
        bounds_v = None
    v = Variable(x.shape, nonpos=True, bounds=bounds_v)

    if x.value is not None:
        t.value = expr.numeric(x.value)
        v.value = np.minimum(x.value - t.value, -1)

    constraints = [sum(exp(v)) == 1, v == x - t]
    return t, constraints
