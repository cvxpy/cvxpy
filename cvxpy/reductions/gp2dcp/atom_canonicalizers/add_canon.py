from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape


def add_canon(expr, args):
    if expr.is_scalar():
        return log_sum_exp(hstack(args)), []
    else:
        rows = []
        r = expr.shape[0]
        c = expr.shape[1] if len(expr.shape) >= 1 else 1
        summands = args
        if c == 1:
            prom_shape = (r, 1)
            summands = [reshape(s, prom_shape) for s in summands]
        for i in range(expr.shape[0]):
            row = []
            for j in range(expr.shape[1]):
                row.append(
                  log_sum_exp(hstack([summand[i, j] for summand in summands])))
            rows.append(row)
        return reshape(bmat(rows), expr.shape), []
