from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def log_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(*shape)
    ones = np.ones(shape)
    # TODO(akshayka): ExpCone requires each of its inputs to be a Variable;
    # is this something that we want to change?
    constraints = [ExpCone(t, ones, x)]
    return t, constraints
