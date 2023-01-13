from cvxpy.expressions.variable import Variable
from cvxpy.utilities import scopes


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
        with scopes.dpp_scope():
            assert rhs.is_affine()
        t = Variable(lhs.shape)
        return op_type(t, rhs), [t == lhs]
    elif rhs.variables():
        with scopes.dpp_scope():
            assert lhs.is_affine()
        t = Variable(rhs.shape)
        return op_type(lhs, t), [t == rhs]

    # Neither side has variables. One side must be affine in parameters.
    lhs_affine = False
    rhs_affine = False
    with scopes.dpp_scope():
        lhs_affine = lhs.is_affine()
        rhs_affine = rhs.is_affine()
    assert lhs_affine or rhs_affine

    if lhs_affine:
        t = Variable(rhs.shape)
        return lhs @ t, [t == rhs]
    else:
        t = Variable(lhs.shape)
        return t @ rhs, [t == lhs]
