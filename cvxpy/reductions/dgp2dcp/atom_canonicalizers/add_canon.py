from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.reshape import reshape


def add_canon(expr, args):
    if expr.is_scalar():
        return log_sum_exp(hstack(args)), []

    rows = []
    summands = [
       promote(s, expr.shape) if s.is_scalar() else s for s in args]
    if len(expr.shape) == 1:
        for i in range(expr.shape[0]):
            row = []
            row.append(
              log_sum_exp(hstack([summand[i] for summand in summands])))
            rows.append(row)
        return reshape(bmat(rows), expr.shape), []
    else:
        for i in range(expr.shape[0]):
            row = []
            for j in range(expr.shape[1]):
                row.append(
                  log_sum_exp(hstack([summand[i, j] for summand in summands])))
            rows.append(row)
        return reshape(bmat(rows), expr.shape), []
