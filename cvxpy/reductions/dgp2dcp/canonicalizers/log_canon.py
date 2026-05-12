from cvxpy.atoms.elementwise.log import log


def log_canon(expr, args, solver_context=None):
    return log(args[0]), []
