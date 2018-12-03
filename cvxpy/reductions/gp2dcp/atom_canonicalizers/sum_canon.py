from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.reductions.gp2dcp.atom_canonicalizers.add_canon import add_canon


def sum_canon(expr, args):
    X = args[0]
    if expr.axis is None:
        x = vec(X)
        summation = sum([xi for xi in x])
        canon, _ = add_canon(summation, summation.args)
        return reshape(canon, expr.shape), []

    if expr.axis == 0:
        X = X.T

    rows = []
    for i in range(X.shape[0]):
        x = vec(X[i])
        summation = sum([xi for xi in x])
        canon, _ = add_canon(summation, summation.args)
        rows.append(canon)
    canon = hstack(rows)
    return reshape(canon, expr.shape), []
