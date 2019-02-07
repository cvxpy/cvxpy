from cvxpy.atoms.affine.diag import diag
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.add_canon import add_canon


def trace_canon(expr, args):
    diag_sum = sum(diag(args[0]))
    return add_canon(diag_sum, diag_sum.args)
