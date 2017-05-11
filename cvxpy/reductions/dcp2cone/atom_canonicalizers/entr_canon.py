from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def entr_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(*shape)
    # -x\log(x) >= t <=> x\exp(t/x) <= 1
    # TODO(akshayka): ExpCone requires each of its inputs to be a Variable;
    # is this something that we want to change?
    ones = Constant(np.ones(shape))
    constraints = [ExpCone(t, x, ones)]
    return t, constraints
