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

from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.atoms.elementwise.log import log
from cvxpy.expressions.variable import Variable

#def abs_canon(expr, args):
#    shape = expr.shape
#    t1 = Variable(shape, bounds = [0, None])
#    if expr.value is not None:
#       #t1.value = np.sqrt(expr.value**2)
#        t1.value = np.abs(args[0].value)
#
#    #return t1, [t1**2 == args[0] ** 2]
#    square_expr = power(args[0], 2)
#    t2, constr_sq = power_canon(square_expr, square_expr.args)
#    return t1, [t1**2 == t2] + constr_sq

def smooth_abs_canon(expr, args):
    shape = expr.shape
    t1 = Variable(shape, bounds = [0, None])
    y = Variable(shape, bounds = [-1.01, 1.01])
    if args[0].value is not None:
        t1.value = np.abs(args[0].value)
        y.value = np.sign(args[0].value)
    
    return t1, [y ** 2 == np.ones(shape), t1 == multiply(y, args[0])]

def approx_abs_canon(expr, args):
    # smooth approximation of abs via
    # \frac{1}{a}\cdot\log\left(e^{ax}+e^{-ax}\right)
    shape = expr.shape
    t1 = Variable(shape, bounds = [0, None])
    a = 1.0
    if args[0].value is not None:
        t1.value = np.abs(args[0].value)
    
    expr = (1/a) * log(exp(a*args[0]) + exp(-a*args[0]))
    return t1, [t1 == expr]
