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
from cvxpy.utilities.bounds import get_expr_bounds

MIN_INIT = 1e-3

# We canonicalize div(f(x), g(x)) as z * y = f(x), y = g(x), y >= 0.
# In other words, it assumes that the denominator is nonnegative.
def div_canon(expr, args):
    
    # raise an error if the denominator is not nonnegative
    lb = args[1].get_bounds()[0]
    if not np.all(lb >= 0):
        raise ValueError("The denominator of a division must be nonnegative. "
                          "Did you forget to specify bounds?")
    
    dim = args[0].shape
    sgn_z = args[0].sign
    bounds_z = get_expr_bounds(expr)

    if sgn_z == 'NONNEGATIVE':
        z = Variable(dim, nonneg=True, bounds=bounds_z)
    elif sgn_z == 'NONPOSITIVE':
        z = Variable(dim, nonpos=True, bounds=bounds_z)
    else:
        z = Variable(dim, bounds=bounds_z)

    bounds_y = get_expr_bounds(args[1])
    y = Variable(args[1].shape, nonneg=True, bounds=bounds_y)

    if args[0].value is not None and args[1].value is not None:
        y.value = np.maximum(args[1].value, MIN_INIT)   
        val = args[0].value / y.value
    
        # dimension hack
        if dim == () and val.shape == (1,):
            z.value = val[0]
        else:
            z.value = val

    return z, [multiply(z, y) == args[0], y == args[1]]