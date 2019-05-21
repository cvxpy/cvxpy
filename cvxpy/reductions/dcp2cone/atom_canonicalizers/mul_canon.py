import contextlib

from cvxpy.expressions.variable import Variable


@contextlib.contextmanager
def _param_vexity_scope(expr):
    """Treats parameters as affine, not constants."""
    for p in expr.parameters():
        p._is_constant = False
    expr._check_is_constant(recompute=True)

    yield

    for p in expr.parameters():
        p._is_constant = True
    expr._check_is_constant(recompute=True)


def mul_canon(expr, args):
    # Only allow param * var (not var * param). Associate right to left.
    # TODO: Only descend if both sides have parameters
    lhs = args[0]
    rhs = args[1]
    if not (lhs.parameters() and rhs.parameters()):
        return expr.copy(args), []

    op_type = type(expr)
    if lhs.variables():
        with _param_vexity_scope(rhs):
            assert rhs.is_affine()
        t = Variable(lhs.shape)
        return op_type(t, rhs), [t == lhs]
    elif rhs.variables():
        with _param_vexity_scope(lhs):
            assert lhs.is_affine()
        t = Variable(rhs.shape)
        return op_type(lhs, t), [t == rhs]

    # Neither side has variables. One side must be affine in parameters.
    lhs_affine = False
    rhs_affine = False
    with _param_vexity_scope(lhs):
        lhs_affine = lhs.is_affine()
    with _param_vexity_scope(rhs):
        rhs_affine = rhs.is_affine()
    assert lhs_affine or rhs_affine

    if lhs_affine:
        t = Variable(rhs.shape)
        return lhs * t, [t == rhs]
    else:
        t = Variable(lhs.shape)
        return t * rhs, [t == lhs]
