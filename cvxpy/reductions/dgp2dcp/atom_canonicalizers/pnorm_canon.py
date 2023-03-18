from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.log_sum_exp import log_sum_exp


def pnorm_canon(expr, args):
    x = args[0]
    p = expr.original_p
    if x.shape == tuple():
        x = promote(p, (1,))
    if expr.axis is None or len(x.shape) == 1:
        x = vec(x)
        return (1.0/p) * log_sum_exp(hstack([xi * p for xi in x])), []

    if expr.axis == 0:
        x = x.T

    rows = []
    for i in range(x.shape[0]):
        row = x[i]
        rows.append((1.0/p) * log_sum_exp(hstack([xi * p for xi in row])))
    return vstack(rows), []
