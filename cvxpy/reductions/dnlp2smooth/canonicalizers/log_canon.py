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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable

# DCED: Without this lower bound the stress test for ML Gaussian non-zero mean fails.
# Perhaps this should be a parameter exposed to the user?
LOWER_BOUND = 1e-5

def log_canon(expr, args):
    t = Variable(args[0].shape, bounds=[LOWER_BOUND, None])

    if args[0].value is not None and np.min(args[0].value) > 5 * LOWER_BOUND:
        t.value = args[0].value
    else:
        t.value = expr.point_in_domain()

    # DCED: introducing an out variable for log works MUCH worse for the 
    # Gaussian ML problem 
    #v = Variable(args[0].size)
    #v.value = np.ones(expr.shape)
    #v.value = np.log(t.value)
    #return v, [v == expr.copy([t]), t == args[0]]

    return expr.copy([t]), [t == args[0]]

# TODO (DCED): On some problems this canonicalization seems to work better.
#              We should investigate this further when we have more benchmarks
#              involving log.
#def log_canon(expr, args):
#    t = Variable(args[0].size)
#    if args[0].value is not None and np.all(args[0].value > 0):
#        t.value = np.log(args[0].value)
#    else:
#        t.value = expr.point_in_domain()

#    return t, [exp(t) == args[0]]