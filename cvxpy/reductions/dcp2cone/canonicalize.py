from cvxpy.reductions.atoms import CANON
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.variable import Constant


def canonicalize_constr(constrs):
    new_constrs = []
    for c in constrs:
        canon_expr, extra_constr = canononicalize_tree(c.expr)
        Constraint(canon_expr, c.set)
        new_constrs += extra_constr
    return new_constrs


def canonicalize_tree(expr):
    canon_args = []
    constrs = []
    for arg in expr.args:
        canon_arg, c = canonicalize_tree(arg)
        canon_args += [canon_arg]
        constrs += c
    canon_expr, c = canonicalize(expr, canon_args)
    constrs += c
    return canon_expr, constrs

 
def canonicalize(expr, args): 
    if expr.is_affine():
        return expr, []
    elif isinstance(expr, Variable):
        return expr, []
    elif isinstance(expr, Constant):
        return expr, []
    else:
        return CANON[type(expr)](expr, args)
