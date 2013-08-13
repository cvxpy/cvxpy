from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

class Minimize(u.Canonicalizable):
    """
    An optimization objective for minimization.
    """
    NAME = "minimize"

    # expr - the expression to minimize.
    def __init__(self, expr):
        self.expr = Expression.cast_to_const(expr)
        # Validate that the objective resolves to a scalar.
        if self.expr.size != (1,1):
            raise Exception("The objective '%s' must resolve to a scalar." 
                            % self.name())
        super(Minimize, self).__init__()

    def __repr__(self):
        return self.name()

    def name(self):
        return ' '.join([self.NAME, self.expr.name()])

    # Create a new objective to handle constants in the original objective.
    # Raise exception if the original objective is not scalar.
    def canonicalize(self):
        obj,constraints = self.expr.canonical_form()
        t,dummy = Variable().canonical_form()
        return (t, constraints + [AffEqConstraint(t, obj)])

    # Objective must be convex.
    def is_dcp(self):
        return self.expr.curvature.is_convex()

    # The value of the objective given the solver primal value.
    def value(self, result):
        return result

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
        return self.expr.curvature.is_concave()

    # The value of the objective given the solver primal value.
    def value(self, result):
        return -result