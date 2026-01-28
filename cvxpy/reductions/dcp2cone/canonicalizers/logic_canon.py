"""
Copyright, the CVXPY authors

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
from cvxpy.utilities.solver_context import SolverInfo


def not_canon(expr, args, solver_context: SolverInfo | None = None):
    return 1 - args[0], []


def and_canon(expr, args, solver_context: SolverInfo | None = None):
    y = Variable(expr.shape, boolean=True)
    constraints = [y <= xi for xi in args]
    constraints.append(y >= sum(args) - (len(args) - 1))
    return y, constraints


def or_canon(expr, args, solver_context: SolverInfo | None = None):
    y = Variable(expr.shape, boolean=True)
    constraints = [y >= xi for xi in args]
    constraints.append(y <= sum(args))
    return y, constraints


def xor_canon(expr, args, solver_context: SolverInfo | None = None):
    n = len(args)
    y = Variable(expr.shape, boolean=True)
    if n == 2:
        x1, x2 = args
        constraints = [
            y <= x1 + x2,
            y >= x1 - x2,
            y >= x2 - x1,
            y <= 2 - x1 - x2,
        ]
    else:
        k = Variable(expr.shape, integer=True)
        constraints = [sum(args) == y + 2 * k, k >= 0, k <= n // 2]
    return y, constraints
