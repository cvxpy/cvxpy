from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.add_canon import add_canon


def sum_canon(expr, args):
    X = args[0]
    if expr.axis is None:
        x = vec(X)
        # sum([expr]) == 0 + expr, which violates DGP, hence the size check.
        summation = sum([xi for xi in x]) if x.size > 1 else x
        canon, _ = add_canon(summation, summation.args)
        return reshape(canon, expr.shape), []

    if expr.axis == 0:
        X = X.T

    rows = []
    for i in range(X.shape[0]):
        x = vec(X[i])
        summation = sum([xi for xi in x]) if x.size > 1 else x
        canon, _ = add_canon(summation, summation.args)
        rows.append(canon)
    canon = hstack(rows)
    return reshape(canon, expr.shape), []
