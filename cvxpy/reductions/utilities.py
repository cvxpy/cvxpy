from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.expressions.constants.constant import Constant


def are_args_affine(constraints):
    return all(arg.is_affine() for constr in constraints
                for arg in constr.args)

def is_stuffed_cone_constraint(constraint):
    """Every constraint formatted by ConeMatrixStuffing is of this form."""
    # TODO(akshayka): Consider coupling this function with the
    # ConeMatrixStuffing class
    for arg in constraint.args:
        if type(arg) == reshape:
            arg = arg.args[0]
        if type(arg) == AddExpression:
            if type(arg.args[0]) != MulExpression:
                return False
            if type(arg.args[0].args[0]) != Constant:
                return False
            if type(arg.args[1]) != Constant:
                return False
        elif type(arg) == MulExpression:
            if type(arg.args[0]) != Constant:
                return False
        else:
            return False
    return True

def is_stuffed_cone_objective(objective):
    """The objective output by ConeMatrixStuffing is of this form."""
    # TODO(akshayka): Consider coupling this function with the
    # ConeMatrixStuffing class
    expr = objective.expr
    return (expr.is_affine()
            and type(expr) == AddExpression
            and len(expr.args) == 2
            and type(expr.args[0]) == MulExpression
            and type(expr.args[1]) == Constant)

def is_stuffed_qp_objective(objective):
    """The objective output by QpMatrixStuffing is of this form."""
    # TODO(akshayka): Consider coupling this function with the
    # QpMatrixStuffing class
    expr = objective.expr
    return (type(expr) == AddExpression
            and len(expr.args) == 2
            and type(expr.args[0]) == QuadForm
            and type(expr.args[1]) == MulExpression
            and expr.args[1].is_affine())
