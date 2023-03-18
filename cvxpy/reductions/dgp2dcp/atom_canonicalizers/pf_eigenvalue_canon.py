from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.mul_canon import mul_canon
from cvxpy.reductions.dgp2dcp.atom_canonicalizers.mulexpression_canon import (
    mulexpression_canon,
)


def pf_eigenvalue_canon(expr, args):
    X = args[0]
    # rho(X) \leq lambda iff there exists v s.t. Xv \leq lambda v
    # v and lambd represent log variables, hence no positivity constraints
    lambd = Variable()
    v = Variable(X.shape[0])
    lhs = matmul(X, v)
    rhs = lambd * v
    lhs, _ = mulexpression_canon(lhs, lhs.args)
    rhs, _ = mul_canon(rhs, rhs.args)
    return lambd, [lhs <= rhs]
