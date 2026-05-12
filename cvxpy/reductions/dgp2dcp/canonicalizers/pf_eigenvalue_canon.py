from cvxpy.atoms.affine.binary_operators import matmul
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dgp2dcp.canonicalizers.mul_canon import mul_canon
from cvxpy.reductions.dgp2dcp.canonicalizers.mulexpression_canon import (
    mulexpression_canon,
)
from cvxpy.utilities.bounds import get_expr_bounds_if_supported
from cvxpy.utilities.values import get_expr_value_if_supported


def pf_eigenvalue_canon(expr, args, solver_context=None):
    X = args[0]
    # rho(X) \leq lambda iff there exists v s.t. Xv \leq lambda v
    # v and lambd represent log variables, hence no positivity constraints
    bounds = get_expr_bounds_if_supported(expr, solver_context)
    lambd = Variable(bounds=bounds)
    value = get_expr_value_if_supported(expr, solver_context)
    if value is not None:
        lambd.value = value
    v = Variable(X.shape[0])
    lhs = matmul(X, v)
    rhs = lambd * v
    lhs, _ = mulexpression_canon(lhs, lhs.args)
    rhs, _ = mul_canon(rhs, rhs.args)
    return lambd, [lhs <= rhs]
