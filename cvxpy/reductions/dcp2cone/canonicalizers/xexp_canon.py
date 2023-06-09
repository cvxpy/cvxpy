"""
Copyright 2022, the CVXPY authors

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

from cvxpy.atoms.elementwise.power import power
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.power_canon import power_canon


def xexp_canon(expr, args):
    x = args[0]
    u = Variable(expr.shape, nonneg=True)
    t = Variable(expr.shape, nonneg=True)
    power_expr = power(x, 2)
    power_obj, constraints = power_canon(power_expr, power_expr.args)

    constraints += [ExpCone(u, x, t),
                    u >= power_obj,
                    x >= 0]
    return t, constraints
