from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def log_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(*shape)
    ones = np.ones(shape)
    constraints = [ExpCone(t, ones, x)]
    return t, constraints
