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

# This file is identical to the dcp2cone version, except that it uses the 
# dnlp2smooth canonicalizer for the power atom. This is necessary.

from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth.canonicalizers.power_canon import power_canon
from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon


def huber_canon(expr, args):
    M = expr.M
    x = args[0]
    shape = expr.shape
    n = Variable(shape)
    s = Variable(shape)

    # n**2 + 2*M*|s|
    power_expr = power(n, 2)
    n2, constr_sq = power_canon(power_expr, power_expr.args)
    abs_expr = abs(s)
    abs_s, constr_abs = abs_canon(abs_expr, abs_expr.args)
    obj = n2 + 2 * M * abs_s

    # x == s + n
    constraints = constr_sq + constr_abs
    constraints.append(x == s + n)
    return obj, constraints
