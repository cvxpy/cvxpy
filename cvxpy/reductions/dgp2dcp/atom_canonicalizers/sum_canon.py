from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.add_canon import add_canon
from cvxpy.reductions.dgp2dcp import util


def sum_canon(expr, args):
    X = args[0]
    if expr.axis is None:
        summation = util.sum(X)
        canon, _ = add_canon(summation, summation.args)
        return reshape(canon, expr.shape), []

    if expr.axis == 0:
        X = X.T

    rows = []
    for i in range(X.shape[0]):
        summation = util.sum(X[i])
        canon, _ = add_canon(summation, summation.args)
        rows.append(canon)
    canon = hstack(rows)
    return reshape(canon, expr.shape), []
