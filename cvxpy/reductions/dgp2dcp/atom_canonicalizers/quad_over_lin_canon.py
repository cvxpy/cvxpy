from cvxpy.atoms.affine.vec import vec


def quad_over_lin_canon(expr, args):
    x = vec(args[0])
    y = args[1]
    numerator = sum(2 * xi for xi in x)
    return numerator - y, []
