def power_canon(expr, args):
    # y = log x; x^p --> exp(y^p) --> p log(exp(y)) = p *  y
    return expr.p * args[0], []
