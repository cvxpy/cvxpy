from cvxpy.atoms.max import max
from cvxpy.reductions.eliminate_pwl.canonicalizers.max_canon import max_canon


def norm_inf_canon(expr, args):
    assert len(args) == 1
    tmp = max(args[0], expr.axis, expr.keepdims)
    return max_canon(tmp, tmp.args)
