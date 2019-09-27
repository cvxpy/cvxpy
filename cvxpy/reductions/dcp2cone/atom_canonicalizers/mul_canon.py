from cvxpy.expressions.constants.parameter import treat_params_as_affine
from cvxpy.expressions.variable import Variable


# TODO(akshayka): expose as a reduction for user's convenience
def mul_canon(expr, args):
    # Only allow param * var (not var * param). Associate right to left.
    # TODO: Only descend if both sides have parameters
    lhs = args[0]
    rhs = args[1]
    if not (lhs.parameters() and rhs.parameters()):
        return expr.copy(args), []

    op_type = type(expr)
    if lhs.variables():
        with treat_params_as_affine(rhs):
            assert rhs.is_affine()
        t = Variable(lhs.shape)
        return op_type(t, rhs), [t == lhs]
    elif rhs.variables():
        with treat_params_as_affine(lhs):
            assert lhs.is_affine()
        t = Variable(rhs.shape)
        return op_type(lhs, t), [t == rhs]

    # Neither side has variables. One side must be affine in parameters.
    lhs_affine = False
    rhs_affine = False
    with treat_params_as_affine(lhs):
        lhs_affine = lhs.is_affine()
    with treat_params_as_affine(rhs):
        rhs_affine = rhs.is_affine()
    assert lhs_affine or rhs_affine

    if lhs_affine:
        t = Variable(rhs.shape)
        return lhs * t, [t == rhs]
    else:
        t = Variable(lhs.shape)
        return t * rhs, [t == lhs]
