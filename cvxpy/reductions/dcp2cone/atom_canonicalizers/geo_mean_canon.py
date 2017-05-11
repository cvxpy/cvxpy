from cvxpy.expressions.variables.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs
import numpy as np


def geo_mean_canon(expr, args):
    x = args[0]
    w = expr.w
    w_dyad = expr.w_dyad
    tree = expr.tree
    shape = expr.shape
    t = Variable(*shape)

    x_list = [x[i] for i in range(len(w))]

    # todo: catch cases where we have (0, 0, 1)?
    # todo: what about curvature case (should be affine) in trivial 
    #       case of (0, 0 , 1)?
    # should this behavior match with what we do in power?
    return t, gm_constrs(t, x_list, w)
