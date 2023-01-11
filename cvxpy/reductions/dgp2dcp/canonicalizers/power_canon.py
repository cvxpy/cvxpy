def power_canon(expr, args):
    # u = log x; x^p --> exp(u^p) --> log(exp(u^p)) = p *  u
    return expr.p * args[0], []
