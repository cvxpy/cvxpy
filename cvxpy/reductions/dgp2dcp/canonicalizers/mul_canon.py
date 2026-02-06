from cvxpy.atoms.affine import add_expr


def mul_canon(expr, args, solver_context=None):
    del expr
    return add_expr.AddExpression(args), []
