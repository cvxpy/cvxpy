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

from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.elementwise.exp import exp
from cvxpy.expressions.variable import Variable


# t = log_sum_exp(x) is equivalent exp(t) = sum(exp(x)), which is 
# equivalent to sum(exp(x - t)) = 1. Now we introduce v = x - t,
# which must be nonpositive.
def log_sum_exp_canon(expr, args):
    x = args[0]
    t = Variable(expr.shape)
    v = Variable(x.shape, nonpos=True)
    
    if x.value is not None:
        t.value = expr.numeric(x.value)
        v.value = x.value - t.value
    else:
        t.value = np.ones(expr.shape)
        v.value = -np.ones(x.shape)

    constraints = [sum(exp(v)) == 1, v == x - t]
    return t, constraints
