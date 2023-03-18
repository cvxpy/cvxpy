from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.atoms.affine.diag import diag
from cvxpy.atoms.one_minus_pos import one_minus_pos
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.mulexpression_canon import (
    mulexpression_canon,
)
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.one_minus_pos_canon import (
    one_minus_pos_canon,
)


def eye_minus_inv_canon(expr, args):
    X = args[0]

    # (I - X)^{-1} \leq T iff there exists 0 <= Y <= T s.t.  YX + I <= Y
    #
    # This function implements the log-log transformation of these constraints
    #
    # We can't use I in DGP, because it has zeros (we'd need to take its log).
    #
    # Instead, the constraint can be written as
    #     diag(diff_pos(Y - YX)) >= 1,
    # or, canonicalized,
    #     lhs_canon >= 0.
    #
    # Here, U = \log Y.
    U = Variable(X.shape)

    # Canonicalization of diag(diff_pos(Y - YX))
    #
    # Note
    #     Y - YX = Y \hadamard (\ones\ones^T - YX/Y) =
    #            = Y \hadamard one_minus_pos(YX/Y),
    # and
    #    Y \hadamard one_minus_pos(YX/Y) canonicalizes to
    #    U + one_minus_pos_canon(YX_canon - Y_canon)
    YX = matmul(U, X)
    YX_canon, _ = mulexpression_canon(YX, YX.args)
    one_minus = one_minus_pos(YX_canon - U)
    canon, _ = one_minus_pos_canon(one_minus, one_minus.args)
    lhs_canon = diag(U + canon)
    return U, [lhs_canon >= 0]
