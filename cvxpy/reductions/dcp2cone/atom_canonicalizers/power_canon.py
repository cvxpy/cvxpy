"""
Copyright 2013 Steven Diamond

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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import (gm_constrs, pow_mid, pow_high,
                                         pow_neg)
import numpy as np


def power_canon(expr, args):
    x = args[0]
    p_orig = expr._p_orig
    max_denom = expr.max_denom

    p, w = None, None
    # compute a rational approximation to p
    # how we convert p to a rational depends on the branch of the function
    if p_orig > 1:
        p, w = pow_high(p_orig, max_denom)
    elif 0 < p_orig < 1:
        p, w = pow_mid(p_orig, max_denom)
    elif p_orig < 0:
        p, w = pow_neg(p_orig, max_denom)

    if p_orig == 1 or p == 1:
        # in case p is a fraction equivalent to 1
        p = 1
        w = None
    if p_orig == 0 or p == 0:
        p = 0
        w = None

    if p == 1:
        return x, []

    shape = expr.shape
    ones = Constant(np.ones(shape))
    if p == 0:
        return ones, []
    else:
        t = Variable(shape)
        # TODO(akshayka): gm_constrs requires each of its inputs to be a Variable;
        # is this something that we want to change?
        if 0 < p < 1:
            return t, gm_constrs(t, [x, ones], w)
        elif p > 1:
            return t, gm_constrs(x, [t, ones], w)
        elif p < 0:
            return t, gm_constrs(ones, [x, t], w)
        else:
            raise NotImplementedError('This power is not yet supported.')
