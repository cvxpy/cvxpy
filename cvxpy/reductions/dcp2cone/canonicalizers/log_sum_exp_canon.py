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

import numpy as np

from cvxpy.atoms import exp, promote, reshape, sum
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.exp_canon import exp_canon
from cvxpy.utilities.bounds import get_expr_bounds_if_supported
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.values import get_expr_value_if_supported


def log_sum_exp_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    keepdims = expr.keepdims
    bounds = get_expr_bounds_if_supported(expr, solver_context)
    t = Variable(shape, bounds=bounds)
    value = get_expr_value_if_supported(expr, solver_context)
    if value is not None:
        t.value = value

    # log(sum(exp(x))) <= t <=> sum(exp(x-t)) <= 1
    if axis is None:  # shape = (1, 1)
        promoted_t = promote(t, x.shape)
    elif axis == 0:  # shape = (1, n)
        promoted_t = Constant(np.ones((x.shape[0], 1))) @ reshape(t, (1,) + x.shape[1:], order="F")
    else:  # shape = (m, 1)
        promoted_t = reshape(t, x.shape[:-1] + (1,), order="F") @ Constant(np.ones((1, x.shape[1])))

    exp_expr = exp(x - promoted_t)
    obj, constraints = exp_canon(exp_expr, exp_expr.args, solver_context=solver_context)
    obj = sum(obj, axis=axis, keepdims=keepdims)
    ones = Constant(np.ones(shape))
    constraints.append(obj <= ones)
    return t, constraints
