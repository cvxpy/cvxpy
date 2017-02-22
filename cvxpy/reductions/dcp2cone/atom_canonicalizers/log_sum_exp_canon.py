from cvxpy.atoms.affine.sum_entries import sum_entries
from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import \
    exp_canon
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def log_sum_exp_canon(expr, args):
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    t = Variable(*shape)

    # log(sum(exp(x))) <= t <=> sum(exp(x-t)) <= 1
    if axis is None:  # shape = (1, 1)
        promoted_t = np.ones(x.shape) * t
    elif axis == 0:  # shape = (1, n)
        promoted_t = np.ones((x.shape[0], 1)) * t
    else:  # shape = (m, 1)
        promoted_t = t * np.ones((1, x.shape[1]))

    exp_expr = x - promoted_t
    obj, constraints = exp_canon(exp_expr, exp_expr.args)
    obj = sum_entries(obj, axis=axis) 
    ones = np.ones(shape)
    constraints.append(obj <= ones)
    return t, constraints
