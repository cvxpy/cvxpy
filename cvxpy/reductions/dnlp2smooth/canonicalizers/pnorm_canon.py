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

from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth.canonicalizers.quad_over_lin_canon import quad_over_lin_canon


def pnorm_canon(expr, args):
    x = args[0]
    p = expr.p
    shape = expr.shape
    t = Variable(shape, nonneg=True)
    # we canonicalize 2-norm as follows:
    # ||x||_2 <= t  <=>  quad_over_lin(x, t) <= t
    if p == 2:
        expr = quad_over_lin(x, t)
        new_expr, constr = quad_over_lin_canon(expr, expr.args)
        return t, constr + [new_expr <= t]
    else:
        raise ValueError("Only p=2 is supported as Pnorm.")
