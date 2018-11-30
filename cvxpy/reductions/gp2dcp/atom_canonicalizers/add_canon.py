from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack


def add_canon(expr, args):
    if expr.is_scalar():
        return log_sum_exp(hstack(args)), []
    else:
        rows = []
        for i in range(expr.shape[0]):
            row = []
            for j in range(expr.shape[1]):
                row.append(
                  log_sum_exp(hstack([summand[i, j] for summand in args])))
            rows.append(row)
        return bmat(rows), []
