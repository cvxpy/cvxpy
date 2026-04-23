def geo_mean_canon(expr, args, solver_context=None):
    out = 0.0
    for x_i, p_i in zip(args[0], expr.p):
        out += p_i * x_i
    return (1 / sum(expr.p)) * out, []
