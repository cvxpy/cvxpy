from cvxpy.constraints.finite_set import FiniteSet


def finite_set_canon(expr, args, solver_context=None):
    ineq_form, id = expr.get_data()
    # Log applied already to set of values (i.e., args[1]).
    return FiniteSet(args[0], args[1], ineq_form, id), []
