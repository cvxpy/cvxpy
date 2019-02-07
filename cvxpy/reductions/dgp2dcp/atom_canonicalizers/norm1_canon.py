from cvxpy.atoms.affine.sum import sum
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.sum_canon import sum_canon


def norm1_canon(expr, args):
    assert len(args) == 1
    tmp = sum(args[0], expr.axis, expr.keepdims)
    return sum_canon(tmp, tmp.args)
