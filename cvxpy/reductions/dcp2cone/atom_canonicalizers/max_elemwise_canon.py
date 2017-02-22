from cvxpy.expressions.variables.variable import Variable


def max_elemwise_canon(expr, args):
    shape = expr.shape
    t = Variable(*shape)
    constraints = [t >= elem for elem in args]
    return t, constraints
