from cvxpy.atoms.affine.diag import diag
from cvxpy.reductions.dgp2dcp.canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp.util import explicit_sum


def trace_canon(expr, args):
    diag_sum = explicit_sum(diag(args[0]))
    return add_canon(diag_sum, diag_sum.args)
