def power_canon(expr, args):
    x = args[0]
    p = expr.p
    w = expr.w

    if p != 2:
        raise ValueError("quadratic form can only have power 2")
    return NotImplemented