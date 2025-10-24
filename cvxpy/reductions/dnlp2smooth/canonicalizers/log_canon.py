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


def is_possible_to_deduce_bounds(expr):

    # we can only deduce bounds if the argument to log is of a * x
    # where a is a constant scalar or vector
    if len(expr.variables()) > 1 or not expr.args[0].is_affine():
        return False, 0
    
    # make sure that it is linear and not just affine 
    x = expr.variables()[0]
    old_x_value = x.value
    x.value = np.zeros(x.shape)
    is_linear = np.all(expr.args[0].value == 0)
    x.value = old_x_value
    if not is_linear:
        return False, 0

    # collect all constants appearing in the expression 
    constants = []
    collect_constants(expr, constants)

    # compute accumulated constant a 
    a = 1
    for constant in constants:
        if constant.ndim > 1:
            return False, 0
        a *= constant.value

    return True, a

def collect_constants(expr, constants):
    if isinstance(expr, Constant):
        constants.append(expr)
    elif hasattr(expr, "args"):
        for subexpr in expr.args:
            collect_constants(subexpr, constants)

# DCED: Without this lower bound the stress test for ML Gaussian non-zero mean fails.
# Perhaps this should be a parameter exposed to the user?
LOWER_BOUND = 1e-5

def log_canon(expr, args):
    t = Variable(args[0].shape, bounds=[LOWER_BOUND, None])

    # if args[0] is a * x for a constant scalar or vector 'a' 
    # and a vector variable 'x', we want to add bounds to x if x
    # does not have any bounds. We also want to initialize x far 
    # from its bounds. 
    possible_to_deduce_bound, a = is_possible_to_deduce_bounds(expr)

    if not possible_to_deduce_bound:
        if args[0].value is not None and np.min(args[0].value) > 5 * LOWER_BOUND:
            t.value = args[0].value
        else:
            t.value = expr.point_in_domain()
    else:
        # at this point we know that args[0] is a * x
        x = expr.variables()[0]  
        if x.value is None:
            x.value = expr.point_in_domain() * np.sign(a)

        lbs = -np.inf * np.ones(x.shape)
        ubs = np.inf * np.ones(x.shape)
        lbs[a > 0] = 0
        ubs[a < 0] = 0

        if x.bounds is not None:
            lbs = np.maximum(lbs, x.bounds[0])
            ubs = np.minimum(ubs, x.bounds[1])

        x.bounds = [lbs, ubs]
        t.value = args[0].value

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