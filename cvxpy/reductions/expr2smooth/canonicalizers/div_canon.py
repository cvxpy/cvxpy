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

from cvxpy.expressions.variable import Variable


# We canonicalize div(x, y) as z * y = x.
def div_canon(expr, args):
    dim = (1, ) if args[0].shape == () else args[0].shape
    sgn_z = args[0].sign

    if sgn_z == 'NONNEGATIVE':
        z = Variable(dim, bounds=[0, None])
    elif sgn_z == 'NONPOSITIVE':
        z = Variable(dim, bounds=[None, 0])
    else:
        z = Variable(dim)
    
    y = Variable(args[1].shape, bounds=[0, None])

    # DCED: perhaps initialize further away from boundary?
    if args[1].value is not None and np.all(args[1].value != 0.0):
        y.value = args[1].value
    else:
        y.value = expr.point_in_domain()

    if args[0].value is not None:
        z.value = np.atleast_1d(args[0].value) / y.value 
    else:
        z.value = expr.point_in_domain()
    return z, [z * y == args[0], y == args[1]]