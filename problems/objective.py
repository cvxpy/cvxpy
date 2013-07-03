class Minimize(object):
    """
    An optimization objective for minimization.
    """
    NAME = "minimize"

    # expr - the expression to minimize.
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return ' '.join([self.NAME, self.expr.name()])

    def linear_ops(self):
        return self.expr.coefficients()

    def variables(self):
        return self.expr.variables()

class Maximize(Minimize): #TODO proper return value
    NAME = "maximize"
    """
    An optimization objective for maximization.
    expr - the expression to maximize.
    """
    def coefficients(self):
        return (-self.expr).coefficients()