from cvxpy.atoms.affine.diag import diag
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp import util


def trace_canon(expr, args):
    diag_sum = util.sum(diag(args[0]))
    return add_canon(diag_sum, diag_sum.args)
