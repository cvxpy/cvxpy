import abc
import cvxpy.settings
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.operators import BinaryOperator
import cvxpy.expressions.types as types

class Constraint(BinaryOperator):
    """
    A constraint on an optimization problem of the form
    affine == affine or affine <= affine.
    Stored internally as affine <=/== 0.
    """
    def __init__(self, lh_exp, rh_exp, value_matrix=intf.DENSE_TARGET):
        self.interface = intf.get_matrix_interface(value_matrix)
        super(Constraint, self).__init__(lh_exp, rh_exp)

    def __repr__(self):
        return self.name()

    def variables(self):
        return (self.lh_exp - self.rh_exp).variables()

    def coefficients(self, interface):
        return (self.lh_exp - self.rh_exp).coefficients(interface)

    # Save the value of the primal variable.
    def save_value(self, value):
        self.dual_value = value

class EqConstraint(Constraint):
    OP_NAME = "=="
    # Both sides must be affine.
    def is_dcp(self):
        return (self.lh_exp - self.rh_exp).curvature.is_affine()

    # TODO expanding equality constraints.
    # Verify doesn't affect dual variables.
    def canonicalize(self, top_level=False):
        return (None, [self])

    @property
    def dual(self):
        return self.dual_value

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
            slack = types.variable()(*obj.size)
            self.slack_equality = (obj + slack == 0)
            constraints += [self.slack_equality, slack >= 0]
        else:
            constraints.append(obj <= 0)
        return (None, constraints)

    # The value of the dual variable.
    @property
    def dual(self):
        return self.slack_equality.dual