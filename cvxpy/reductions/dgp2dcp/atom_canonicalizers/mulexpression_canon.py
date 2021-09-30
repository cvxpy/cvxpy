"""Canonicalization for matrix multiplication."""

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.log_sum_exp import log_sum_exp
from cvxpy.utilities.shape import mul_shapes_promote


def mulexpression_canon(expr, args):
    lhs = args[0]
    rhs = args[1]
    lhs_shape, rhs_shape, _ = mul_shapes_promote(lhs.shape, rhs.shape)
    lhs = reshape(lhs, lhs_shape)
    rhs = reshape(rhs, rhs_shape)
    rows = []
    # TODO(akshayka): Parallelize this for large matrices.
    for i in range(lhs.shape[0]):
        row = []
        for j in range(rhs.shape[1]):
            arr = hstack([lhs[i, k] + rhs[k, j] for k in range(lhs.shape[1])])
            row.append(log_sum_exp(arr))
        rows.append(row)
    mat = bmat(rows)
    if mat.shape != expr.shape:
        mat = reshape(mat, expr.shape)
    return mat, []
