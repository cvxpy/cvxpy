import cvxpy
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.affine import AffEqConstraint
from card_variable import CardVariable

class max_card(Constraint):
    """ Imposes a cardinality constraint on an affine expression. """
    # exp - the expression to be constrainted.
    # k - the maximum cardinality.
    def __init__(self, exp, k):
        self.exp = exp
        self.card_var = CardVariable(k, *exp.size)

    # Non-DCP
    def is_dcp(self):
        return False

    # Return the constraint self.card_var == self.exp
    def canonicalize(self):
        return (None,[AffEqConstraint(self.card_var, self.exp)])


class card(object):
    """ 
    A wrapper to convert card(exp) <=/== k to max_card(exp,k).
    """
    def __init__(self, exp):
        self.exp = exp
        if not self.exp.curvature.is_affine():
            raise Exception("Cannot evaluate the cardinality of non-affine expressions.")

    def __eq__(self, k):
        return max_card(self.exp, k)

    def __le__(self, k):
        return max_card(self.exp, k)