from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import \
    exp_canon
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def max_entries_canon(expr, args):
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    t = Variable(*shape)

    if axis is None:  # shape = (1, 1)
        promoted_t = np.ones(x.shape) * t
    elif axis == 0:  # shape = (1, n)
        promoted_t = np.ones((x.shape[0], 1)) * t
    else:  # shape = (m, 1)
        promoted_t = t * np.ones((1, x.shape[1]))

    constraints = [x <= promoted_t]
    return t, constraints
