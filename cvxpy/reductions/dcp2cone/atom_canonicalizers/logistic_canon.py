from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import \
    exp_canon
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def logistic_canon(expr, args):
    x = args[0]
    shape = expr.shape
    # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
    t0 = Variable(*shape)
    t1, constr1 = exp_canon(expr, [-t0])
    t2, constr2 = exp_canon(expr, [x - t0])
    ones = Constant(np.ones(shape))
    constraints = constr1 + constr2 + [t1 + t2 <= ones]
    return t0, constraints
