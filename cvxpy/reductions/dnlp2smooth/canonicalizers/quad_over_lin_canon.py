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

from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dnlp2smooth.canonicalizers.power_canon import power_canon


def quad_over_lin_canon(expr, args):
    """
    Canonicalize a quadratic over linear expression.
    If the denominator is constant, we can use the power canonicalizer.
    Otherwise, we introduce new variables for the numerator and denominator.
    """
    if args[1].is_constant():
        expr = power(args[0], 2)
        var, constr = power_canon(expr, expr.args)
        summation = Sum(var)
        return 1/args[1].value * summation, constr
    else:
        t1 = args[0]
        t2 = args[1]
        constraints = []
        if not isinstance(t1, Variable):
            t1 = Variable(t1.shape)
            constraints += [t1 == args[0]]
            t1.value = args[0].value
        # always introduce a new variable for the denominator
        # so that we can initialize it to 1 (point in domain)
        t2 = Variable(t2.shape)
        constraints += [t2 == args[1]]
        t2.value = np.ones(t2.shape)
        return expr.copy([t1, t2]), constraints
