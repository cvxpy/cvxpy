from cvxpy.reductions.dgp2dcp.util import explicit_sum


def quad_over_lin_canon(expr, args):
    numerator = explicit_sum(2 * args[0])
    return numerator - args[1], []
