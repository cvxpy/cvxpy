"""
Copyright 2021 The CVXPY Developers

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

from cvxpy.atoms.affine.promote import promote
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.bounds import get_expr_bounds_if_supported
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.values import get_expr_value_if_supported


def rel_entr_canon(expr, args, solver_context: SolverInfo | None = None):
    shape = expr.shape
    x = promote(args[0], shape)
    y = promote(args[1], shape)
    # Return is -t, so t_bounds = negated expr_bounds
    expr_bounds = get_expr_bounds_if_supported(expr, solver_context)
    t_bounds = None
    if expr_bounds is not None:
        t_bounds = [-expr_bounds[1], -expr_bounds[0]]
    t = Variable(shape, bounds=t_bounds)
    expr_value = get_expr_value_if_supported(expr, solver_context)
    if expr_value is not None:
        t.value = -expr_value
    constraints = [ExpCone(t, x, y)]
    obj = -t
    return obj, constraints
