import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.variable import Variable

class Minimize(object):
    """
    An optimization objective for minimization.
    """
    NAME = "minimize"

    # expr - the expression to minimize.
    def __init__(self, expr):
        self.expr = Variable.cast_to_const(expr)

    def __repr__(self):
        return self.name()

    def name(self):
        return ' '.join([self.NAME, self.expr.name()])

    # Create a new objective to handle constants in the original objective.
    # Raise exception if the original objective is not scalar.
    def canonicalize(self):
        if self.expr.size() != (1,1):
            raise Exception("The objective '%s' must resolve to a scalar." 
                            % self.name())
        obj,constraints = self.expr.canonicalize()
        t = Variable()
        return (t, constraints + [t == obj])

    # Objective must be convex.
    def is_dcp(self):
        return self.expr.curvature().is_convex()

    # The value of the objective, taken from the solver results.
    def value(self, results):
        return results['primal objective']

class Maximize(Minimize):
    NAME = "maximize"
    """
    An optimization objective for maximization.
    """
    def canonicalize(self):
        obj,constraints = super(Maximize, self).canonicalize()
        return (-obj, constraints)

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