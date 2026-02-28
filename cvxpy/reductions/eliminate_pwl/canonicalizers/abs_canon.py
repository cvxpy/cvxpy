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


from cvxpy.expressions.variable import Variable
from cvxpy.utilities.bounds import get_expr_bounds_if_supported
from cvxpy.utilities.solver_context import SolverInfo


def abs_canon(expr, args, solver_context: SolverInfo | None = None):
    x = args[0]
    bounds = get_expr_bounds_if_supported(expr, solver_context)
    t = Variable(expr.shape, bounds=bounds)
    constraints = [t >= x, t >= -x]

    # for DNLP we must initialize the new variable (DNLP guarantees that 
    # x.value will be set when this function is called)
    if expr.value is not None:
        t.value = expr.value

    return t, constraints
