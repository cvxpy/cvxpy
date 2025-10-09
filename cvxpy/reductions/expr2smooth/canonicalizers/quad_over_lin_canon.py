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
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.elementwise.power import power
from cvxpy.reductions.expr2smooth.canonicalizers.power_canon import power_canon


def quad_over_lin_canon(expr, args):
    """
    Canonicalize a quadratic over linear expression.
    We use the base atoms power and div to do so.
    """
    if args[1].is_constant() and args[1].value == 1:
        expr = power(args[0], 2)
        var, constr = power_canon(expr, expr.args)
        summation = Sum(var)
        return summation, constr
    else:
        assert(False)
