from cvxpy.expressions.variables.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs
import numpy as np


def geo_mean_canon(expr, args):
    x = args[0]
    p = expr.p
    w = expr.w

    if p == 1:
        return x, []

    shape = expr.shape
    ones = np.ones(shape)
    if p == 0:
        return ones, []
    else:
        t = Variable(*shape)
        if 0 < p < 1:
            return t, gm_constrs(t, [x, ones], w)
        elif p > 1:
            return t, gm_constrs(x, [t, ones], w)
        elif p < 0:
            return t, gm_constrs(ones, [x, t], w)
        else:
            raise NotImplementedError('This power is not yet supported.')
