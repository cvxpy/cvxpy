import cvxpy
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.affine import AffEqConstraint
from card_variable import CardVariable

class max_card(Constraint):
    """ Imposes a cardinality constraint on an affine expression. """
    # exp - the expression to be constrainted.
    # k - the maximum cardinality.
    def __init__(self, exp, k):
        if exp.curvature.is_affine() and not exp.curvature.is_constant():
            self.card_var = CardVariable(k, *exp.size)
            self.exp = exp
        else:
            raise Exception("Non-affine expression cannot have a cardinality constraint.")

    # Non-DCP
    def is_dcp(self):
        return False

    # Return the constraint self.card_var == self.exp
    def canonicalize(self):
        return (None,[AffEqConstraint(self.card_var, self.exp)])