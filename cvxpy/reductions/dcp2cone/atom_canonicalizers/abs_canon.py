from cvxpy.expressions.variables.variable import Variable

def abs_canon(expr, args):
    x = args[0]
    t = Variable(x.size[0], x.size[1])
    constraints = [t >= x, t >= -x]
    return t, constraints
