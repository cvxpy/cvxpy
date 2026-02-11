"""
NumPy-like block matrix constructor for CVXPY.
"""

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.vstack import vstack


def _atleast_2d(expr):
    expr = AffAtom.cast_to_const(expr)

    # Promote scalars to (1,1)
    if expr.ndim == 0:
        return expr.reshape((1, 1), order="F")

    # Promote 1D to row vector (1,n) like numpy.block
    if expr.ndim == 1:
        return expr.reshape((1, expr.shape[0]), order="F")

    return expr


def block(block_lists):
    """
    NumPy-like block matrix assembly.

    Scalars become (1,1).
    1D arrays become row vectors (1,n).
    """

    promoted = []
    for row in block_lists:
        promoted.append([_atleast_2d(elem) for elem in row])

    row_blocks = [hstack(row) for row in promoted]
    return vstack(row_blocks)
