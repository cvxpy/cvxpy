from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.add_canon import add_canon


def sum_canon(expr, args):
    X = args[0]
    if expr.axis is None:
        x = vec(X)
        # the Python `sum` function is a reduction with initial value 0.0,
        # resulting in a non-DGP expression
        summation = x[0]
        for xi in x[1:]:
            summation += xi
        canon, _ = add_canon(summation, summation.args)
        return reshape(canon, expr.shape), []

    if expr.axis == 0:
        X = X.T

    rows = []
    for i in range(X.shape[0]):
        x = vec(X[i])
        summation = x[0]
        for xi in x[1:]:
            summation += xi
        canon, _ = add_canon(summation, summation.args)
        rows.append(canon)
    canon = hstack(rows)
    return reshape(canon, expr.shape), []
