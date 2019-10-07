from cvxpy.atoms.affine.vec import vec


# the Python `sum` function is a reduction with initial value 0.0,
# resulting in a non-DGP expression
def explicit_sum(expr):
    x = vec(expr)
    summation = x[0]
    for xi in x[1:]:
        summation += xi
    return summation
