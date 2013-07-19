import cvxpy.interface.matrices as intf

class Minimize(object):
    """
    An optimization objective for minimization.
    """
    NAME = "minimize"

    # expr - the expression to minimize.
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return self.name()

    def name(self):
        return ' '.join([self.NAME, self.expr.name()])

    # def size(self):
    #     if self.expr.size() != (1,1):
    #         raise Exception("The objective '%s' must resolve to a scalar." 
    #                         % self.name())
    #     return self.expr.size()

    def canonicalize(self):
        return self.expr.canonicalize()

    # Objective must be convex.
    def is_dcp(self):
        return self.expr.curvature().is_convex()

    # The value of the objective, taken from the solver results.
    def value(self, results):
        return results['primal objective'] #+ self.constant

class Maximize(Minimize):
    NAME = "maximize"
    """
    An optimization objective for maximization.
    """
    # # Store the constant term.
    # def coefficients(self):
    #     coeff_dict = (-self.expr).coefficients()
    #     self.constant = self.expr.constant(coeff_dict)
    #     return coeff_dict

    def canonicalize(self):
        return (-self.expr).canonicalize()

    # Objective must be concave.
    def is_dcp(self):
        return self.expr.curvature().is_concave()

    # The value of the objective, taken from the solver results.
    def value(self, results):
        return -super(Maximize, self).value(results)


def minimize(expr):
    return Minimize(expr)

def maximize(expr):
    return Maximize(expr)