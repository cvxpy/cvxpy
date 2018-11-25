def div_canon(expr, args):
    del expr
    # x / y == x * y**(-1).
    return args[0] - args[1], []
