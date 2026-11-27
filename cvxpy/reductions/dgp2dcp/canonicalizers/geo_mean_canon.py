from cvxpy.atoms.affine.reshape import reshape


def geo_mean_canon(expr, args):
    # Reduced entries along axis 0, zero-weight ones already dropped, so the
    # weights line up with the rows whatever the axis is.
    x = expr._aligned_arg(args[0])
    out = 0.0
    for i, p_i in enumerate(expr.p):
        out += p_i * x[i]
    out = (1 / sum(expr.p)) * out
    if out.shape != expr.shape:
        out = reshape(out, expr.shape, order='F')
    return out, []
