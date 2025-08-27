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


def collect_constant_and_variable(expr, constants, variable):
    if isinstance(expr, Constant):
        constants.append(expr)
    elif isinstance(expr, Variable):
        variable.append(expr)
    elif hasattr(expr, "args"):
        for subexpr in expr.args:
            collect_constant_and_variable(subexpr, constants, variable)

    assert(len(variable) <= 1)

# DCED: Without this lower bound the stress test for ML Gaussian non-zero mean fails.
# Perhaps this should be a parameter exposed to the user?
LOWER_BOUND = 1e-5

def log_canon(expr, args):
    t = Variable(args[0].size, bounds=[LOWER_BOUND, None], name='t')

    # DCED: if args[0] is a * x for a constant scalar or vector 'a' 
    # and a vector variable 'x', we want to add bounds to x if x
    # does not have any bounds. We also want to initialize x far 
    # from its bounds. 
    constants, variable = [], []
    collect_constant_and_variable(args[0], constants, variable) 
    a = 1
    is_special_case = True
    for constant in constants:
        if len(constant.shape) == 2:
            is_special_case = False
            break
        else:
            a *= constant.value

    if not is_special_case:
        if args[0].value is not None and np.all(args[0].value > 0):
            t.value = args[0].value
        else:
            t.value = expr.point_in_domain()
    else:  
        if variable[0].value is None:
            variable[0].value = expr.point_in_domain() * np.sign(a)
        
        lbs = -np.inf * np.ones(args[0].size)
        ubs = np.inf * np.ones(args[0].size)
        lbs[a > 0] = 0
        ubs[a < 0] = 0
        variable[0].bounds = [lbs, ubs]        
        assert(args[0].value is not None and np.all(args[0].value > 0.0))
        t.value = args[0].value

    return expr.copy([t]), [t==args[0]]
