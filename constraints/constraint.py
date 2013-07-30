import abc
import cvxpy.settings
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.operators import BinaryOperator
import cvxpy.expressions.variable #TODO fix import circle

class Constraint(BinaryOperator):
    """
    A constraint on an optimization problem of the form
    affine == affine or affine <= affine.
    Stored internally as affine <=/== 0.
    """
    def __repr__(self):
        return self.name()

    def variables(self):
        return (self.lh_exp - self.rh_exp).variables()

class EqConstraint(Constraint):
    OP_NAME = "=="
    # Both sides must be affine.
    def is_dcp(self):
        return (self.lh_exp - self.rh_exp).curvature.is_affine()

    # TODO expanding equality constraints.
    # Verify doesn't affect dual variables.
    def canonicalize(self, top_level=False):
        return (None, [self])

class LeqConstraint(Constraint):
    OP_NAME = "<="
    # Left hand expression must be convex and right hand must be concave.
    def is_dcp(self):
        return (self.lh_exp - self.rh_exp).curvature.is_convex()

    # Canonicalize the expression in the constraint and
    # add a new constraint with the expression objective.
    # top_level - is the constraint one of those defined for the problem,
    #             or was it generated during canonicalization?
    def canonicalize(self, top_level=False):
        obj,constraints = (self.lh_exp - self.rh_exp).canonicalize()
        if top_level: # Replace inequality with an equality with slack.
            slack = cvxpy.expressions.variable.Variable(*obj.size)
            self.slack_equality = (obj + slack == 0)
            constraints += [self.slack_equality, slack >= 0]
        else:
            constraints.append(obj <= 0)
        return (None, constraints)

    # The value of the dual variable.
    @property
    def value(self):
        return self.slack_equality.value