from numpy import ones

from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variables import Variable

def power_canon(expr, args):
    affine_expr = args[0]
    p = expr.p
    if p == 0:
        return ones(affine_expr.shape), []
    elif p == 1:
        return affine_expr, []
    elif p == 2:
        t = Variable(*affine_expr.shape)
        return SymbolicQuadForm(t, None, expr), [affine_expr == t]
    raise ValueError("quadratic form can only have power 2")
