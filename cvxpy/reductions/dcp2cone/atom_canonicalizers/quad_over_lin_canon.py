from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variables.variable import Variable


def quad_over_lin_canon(expr, args):
    # quad_over_lin := sum_{ij} X^2_{ij} / y
    x = args[0]
    y = args[1]
    shape = expr.shape
    # precondition: shape == (1,)
    t = Variable(*shape)
    # (y+t, y-t, 2*x) must lie in the second-order cone,
    # where y+t is the scalar part of the second-order
    # cone constraint.
    constraints = [SOC(
                        t=y+t,
                        X=vstack(y-t, reshape(2*x, (x.shape[0] * x.shape[1], 1))), axis=0
                        ), y >= 0]
    return t, constraints
