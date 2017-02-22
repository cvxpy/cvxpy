from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def exp_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(*shape)
    ones = np.ones(shape)
    constraints = [ExpCone(x, ones, t)]
    return t, constraints
