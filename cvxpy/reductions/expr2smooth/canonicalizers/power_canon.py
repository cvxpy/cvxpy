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
    w = expr.w

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []
    else:
        t = Variable(shape)
        if 0 < p < 1:
            return t, [t**(1/p) == x, t >= 0]
        elif p > 1:
            return x**p, []
        else:  # p < 0
            raise ValueError(
                "Power canonicalization does not support negative powers."
            )
