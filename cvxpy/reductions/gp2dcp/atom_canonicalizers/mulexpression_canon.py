"""Canonicalization for matrix multiplication."""

from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack


def mulexpression_canon(expr, args):
    del expr
    lhs = args[0]
    rhs = args[1]
    rows = []
    # TODO(akshayka): Parallelize this for large matrices.
    for i in range(lhs.shape[0]):
        row = []
        for j in range(rhs.shape[1]):
            arr = hstack([lhs[i, k] + rhs[k, j] for k in range(lhs.shape[1])])
            row.append(log_sum_exp(arr))
        rows.append(row)
    return bmat(rows), []
