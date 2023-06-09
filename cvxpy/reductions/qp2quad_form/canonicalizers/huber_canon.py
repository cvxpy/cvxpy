"""
Copyright 2018 CVXPY.

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

from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
from cvxpy.reductions.qp2quad_form.canonicalizers.power_canon import (
    power_canon,
)


def huber_canon(expr, args):
    M = expr.M
    x = args[0]
    shape = expr.shape
    n = Variable(shape)
    s = Variable(shape)

    # n**2 + 2*M*|s|
    # TODO(akshayka): Make use of recursion inherent to canonicalization
    # process and just return a power / abs expressions for readability sake
    power_expr = power(n, 2)
    n2, constr_sq = power_canon(power_expr, power_expr.args)
    abs_expr = abs(s)
    abs_s, constr_abs = abs_canon(abs_expr, abs_expr.args)
    obj = n2 + 2 * M * abs_s

    constraints = constr_sq + constr_abs
    constraints.append(x == s + n)

    return obj, constraints
