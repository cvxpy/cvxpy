from cvxpy.atoms.affine import add_expr


def mul_canon(expr, args):
    del expr
    return add_expr.AddExpression(args), []
