from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variables.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs
import numpy as np


def huber_canon(expr, args):
    M = expr.M
    x = args[0]
    shape = expr.shape
    n = Variable(*shape)
    s = Variable(*shape)

    # n**2 + 2*M*|s|
    power_expr = power(n, 2)
    n2, constr_sq = power_canon(power_exp, power_exp.args)
    abs_expr = abs(s)
    abs_s, constr_abs = abs_canon(abs_expr, abs.args)
    obj = n2 + 2 * M * abs_s

    # x == s + n
    constraints = constr_sq + constr_abs
    constraints.append(x == s + n)
    return obj, constraints
