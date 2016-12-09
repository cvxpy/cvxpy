from cvxpy.reductions.dcp2cone.atom_canonicalizers import CANON_METHODS
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Constant


# TODO this assumes all possible constraint sets are cones:
def canonicalize_constr(constr):
    arg_exprs = []
    constrs = []
    for a in constr.args:
        e, c = canonicalize_tree(a)
        constrs += c
        arg_exprs += [e]
    # Feed the linear expressions into a constraint of the same type (assumed a cone):
    constr = type(constr)(*arg_exprs)
    return constr, constrs


def canonicalize_tree(expr):
    canon_args = []
    constrs = []
    for arg in expr.args:
        canon_arg, c = canonicalize_tree(arg)
        canon_args += [canon_arg]
        constrs += c
    canon_expr, c = canonicalize_expr(expr, canon_args)
    constrs += c
    return canon_expr, constrs


def canonicalize_expr(expr, args): 
    if isinstance(expr, Variable):
        return expr, []
    elif isinstance(expr, Constant):
        return expr, []
    elif expr.is_atom_convex and expr.is_atom_concave:
        return expr, []
    else:
        return CANON_METHODS[type(expr)](expr, args)
