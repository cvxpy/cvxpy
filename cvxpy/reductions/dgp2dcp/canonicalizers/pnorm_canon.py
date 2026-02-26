import math

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.transpose import transpose
from cvxpy.atoms.affine.vec import vec
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.log_sum_exp import log_sum_exp


def pnorm_canon(expr, args):
    x = args[0]
    p = expr.original_p
    if x.shape == tuple():
        x = promote(x, (1,))
    if expr.axis is None or len(x.shape) == 1:
        x = vec(x, order='F')
        return (1.0/p) * log_sum_exp(hstack([xi * p for xi in x])), []

    axis = expr.axis
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    ndim = len(x.shape)

    # Permute so non-reduce axes come first, reduce axes come last
    keep = [i for i in range(ndim) if i not in axes]
    perm = keep + list(axes)
    x_perm = transpose(x, axes=perm) if perm != list(range(ndim)) else x

    # Reshape to 2D: (n_output, n_reduce)
    n_output = math.prod(x.shape[i] for i in keep)
    n_reduce = math.prod(x.shape[i] for i in axes)
    x_2d = reshape(x_perm, (n_output, n_reduce), order='C')

    rows = []
    for i in range(n_output):
        row = x_2d[i]
        rows.append((1.0/p) * log_sum_exp(hstack([xi * p for xi in row])))
    return reshape(vstack(rows), expr.shape, order='F'), []
