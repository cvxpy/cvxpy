from cvxpy.atoms.elementwise.log import log


def log_canon(expr, args):
    return log(args[0]), []
