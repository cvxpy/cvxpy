from cvxpy.reductions.dgp2dcp import util


def quad_over_lin_canon(expr, args):
    numerator = util.sum(2 * args[0])
    return numerator - args[1], []
