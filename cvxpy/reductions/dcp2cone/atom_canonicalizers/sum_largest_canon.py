from cvxpy.expressions.variables.variable import Variable
import numpy as np


def sum_largest_canon(expr, args):
    x = args[0]
    k = expr.k
    shape = expr.shape

    # min sum_entries(t) + kq
    # s.t. x <= t + q
    #      0 <= t
    t = Variable(*shape)
    q = Variable(1)
    obj = sum_entries(t) + k*q
    constraints = [x <= t + q, t >= 0]
    return obj, constraints
