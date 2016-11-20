from cvxpy.expressions.variables.variable import Variable

def abs_canon(expr, args):
    x = args
    t = Variable(x.size)
    constraints = [t >= x, t >= -x]
    return t, constraints
