from cvxpy.reductions.dgp2dcp.canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp.util import explicit_sum


def quad_over_lin_canon(expr, args, solver_context=None):
    summed = explicit_sum(2 * args[0])
    numerator, _ = add_canon(summed, summed.args)
    return numerator - args[1], []
