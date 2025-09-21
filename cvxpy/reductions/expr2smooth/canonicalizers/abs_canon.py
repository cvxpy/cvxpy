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

# This formulation is LICQ-friendly but might introduce spurious KKT points.
def abs_canon(expr, args):
    shape = expr.shape
    t1 = Variable(shape, bounds = [0, None])
    y = Variable(shape, bounds = [-1.01, 1.01])
    if args[0].value is not None:
        t1.value = np.abs(args[0].value)
        y.value = np.sign(args[0].value)
    
    return t1, [y ** 2 == np.ones(shape), t1 == multiply(y, args[0])]
