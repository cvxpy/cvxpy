class Minimize(object):
    NAME = "minimize"
    """
    An optimization objective for minimization.
    expr - the expression to minimize.
    """
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return ' '.join([self.NAME, self.expr.name()])

    def coefficients(self):
        return self.expr.coefficients()

    def shape(self):
        return self.expr.shape()

    def variables(self):
        return self.expr.variables()

    # Raises an error if expr does not resolve to a scalar.
    def validate(self):
        if not self.expr.shape().is_scalar():
            raise Exception( ("Cannot %s %s " % (self.NAME, self.expr.name()),
                              "because it does not resolve to a scalar.") )

class Maximize(Minimize): #TODO proper return value
    NAME = "maximize"
    """
    An optimization objective for maximization.
    expr - the expression to maximize.
    """
    def coefficients(self):
        return (-self.expr).coefficients()