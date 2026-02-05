def power_canon(expr, args, solver_context=None):
    # u = log x; x^p --> exp(u^p) --> log(exp(u^p)) = p *  u
    return expr.p * args[0], []
