def gmatmul_canon(expr, args, solver_context=None):
    return expr.A @ args[0], []
