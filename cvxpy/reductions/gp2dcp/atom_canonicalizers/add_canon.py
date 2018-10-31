from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.affine.hstack import hstack


def add_canon(expr, args):
    del expr
    return log_sum_exp(hstack(args)), []
