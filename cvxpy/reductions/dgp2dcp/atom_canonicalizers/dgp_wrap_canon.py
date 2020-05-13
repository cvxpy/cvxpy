def dgp_wrap_canon(expr, args):
    """DGP wrapped functions are 'unwrapped' by dgp2dcp.
    """
    func = expr.function
    del expr
    return func(*args)
