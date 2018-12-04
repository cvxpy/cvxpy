from cvxpy.atoms.elementwise.exp import exp


def exp_canon(expr, args):
    del expr
    return exp(args[0]), []
