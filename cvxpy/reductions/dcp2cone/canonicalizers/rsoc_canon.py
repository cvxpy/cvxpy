"""
Canonicalizer for the Rotated Second Order Cone (RSOC) constraint.

Converts RSOC(x, y, z) into:
    SOC(y + z, [2x; y - z])   (i.e., ||[2x; y-z]||_2 <= y + z)
    y >= 0
    z >= 0

This encodes: yz >= ||x||^2, y >= 0, z >= 0
"""

from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.constraints.second_order import SOC


def rsoc_canon(expr, args, **kwargs):
    x, y, z = args

    # Flatten all to 1D and concatenate for the SOC body
    x_flat = reshape(x, (x.size,), order='C')
    diff = reshape(y - z, (1,), order='C')

    soc_body = hstack([2 * x_flat, diff])
    soc_scalar = reshape(y + z, (1,), order='C')

    soc_con = SOC(soc_scalar, soc_body)

    nonneg_y = y >= 0
    nonneg_z = z >= 0

    return soc_con, [nonneg_y, nonneg_z]
