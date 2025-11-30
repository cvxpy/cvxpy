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


def power_canon(expr, args):
    x = args[0]
    p = expr.p_rational 
    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []
    elif p == 1:
        return x, []
    elif p == 0.5:
        t = Variable(shape, nonneg=True)

        if x.value is not None:
            t.value = x.value
        else:
            t.value = expr.point_in_domain()

        return expr.copy([t]), [t == x]
    elif isinstance(p, int) and p > 1:
        if isinstance(x, Variable):
            return expr.copy(args), []

        t = Variable(shape)
        if x.value is not None:
            t.value = x.value
        else:
            t.value = expr.point_in_domain()

        return expr.copy([t]), [t == x]
    else:
        raise NotImplementedError(f'The power {p} is not yet supported.')
