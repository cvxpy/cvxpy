from cvxpy.atoms.elementwise.exp import exp


def xexp_canon(expr, args, solver_context=None):
    del expr
    return args[0] + exp(args[0]), []
