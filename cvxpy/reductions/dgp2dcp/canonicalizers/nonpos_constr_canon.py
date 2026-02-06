from cvxpy.constraints.nonpos import NonPos


def nonpos_constr_canon(expr, args, solver_context=None):
    assert len(args) == 2
    return NonPos(args[0] - args[1], constr_id=expr.id), []
