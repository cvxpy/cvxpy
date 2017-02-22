from cvxpy.expressions.variables.variable import Variable
from cvxpy.reductions.dcp2cone.atom_canonicalizers import log_canon


def log1p_canon(expr, args):
    return log_canon(expr, [args[0] + 1])
