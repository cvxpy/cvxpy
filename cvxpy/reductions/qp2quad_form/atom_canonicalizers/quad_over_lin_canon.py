from cvxpy.expressions.variables import Variable
from cvxpy.atoms.quad_form import SymbolicQuadForm
from numpy import eye


def quad_over_lin_canon(expr, args):
    affine_expr = args[0]
    y = args[1]
    t = Variable(*affine_expr.shape)
    return SymbolicQuadForm(t, eye(affine_expr.size)/y, expr), [affine_expr == t]
