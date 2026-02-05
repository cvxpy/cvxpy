from cvxpy.atoms.elementwise.exp import exp


def exp_canon(expr, args, solver_context=None):
    del expr
    return exp(args[0]), []
