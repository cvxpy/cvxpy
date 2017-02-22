from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.variables.variable import Variable
import numpy as np


def kl_div_canon(expr, args):
    x = args[0]
    y = args[1]
    shape = expr.shape
    t = Variable(*shape)
    constraints = [ExpCone(t, x, y), y >= 0]
    obj = y - x - t
    return obj, constraints
