from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.gp2dcp.atom_canonicalizers.mulexpression_canon import mulexpression_canon
import numpy as np


def eye_minus_inv_canon(expr, args):
    X = args[0]
    # (I - X)^{-1} \leq T iff there exists 0 <= Y <= T s.t.  YX + Y <= Y
    Y = Variable(X.shape, pos=True)
    prod = matmul(Y, X)
    lhs, _ = mulexpression_canon(prod, prod.args)
    lhs += np.eye(prod.shape[0])
    return Y, [lhs <= Y]
