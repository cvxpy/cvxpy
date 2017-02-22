from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variables.variable import Variable


def quad_over_lin_canon(expr, args):
    x = args[0]
    y = args[1]
    shape = expr.shape
    # precondition: shape == (1,)
    t = Variable(*shape)
    constraints = [SOC(y+t, y-t, 2*x), y >= 0]
    return t, constraints
