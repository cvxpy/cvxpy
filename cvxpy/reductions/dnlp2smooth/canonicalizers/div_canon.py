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

MIN_BOUND = 1e-4

# We canonicalize div(f(x), g(x)) as z * y = f(x), y = g(x), y >= 0.
# In other words, it assumes that the denominator is nonnegative.
# TODO (DCED):
#  1.  is it necessary to add the bound y >= 0? Does it help in
#      terms of robustness?
def div_canon(expr, args):
    dim = args[0].shape 
    sgn_z = args[0].sign

    if sgn_z == 'NONNEGATIVE':
        z = Variable(dim, bounds=[0, None])
    elif sgn_z == 'NONPOSITIVE':
        z = Variable(dim, bounds=[None, 0])
    else:
        z = Variable(dim)
    
    y = Variable(args[1].shape, bounds=[0, None])

    if args[1].value is not None and np.min(args[1].value) > MIN_BOUND:
        y.value = args[1].value
    else:
        y.value = expr.point_in_domain()

    if args[0].value is not None:
        val = args[0].value / y.value    
    else:
        val = expr.point_in_domain()
    
    # dimension hack
    if dim == () and val.shape == (1,):
        z.value = val[0]
    else:
        z.value = val

    return z, [multiply(z, y) == args[0], y == args[1]]