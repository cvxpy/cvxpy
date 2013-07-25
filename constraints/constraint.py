import abc
import cvxpy.settings
import cvxpy.interface.matrix_utilities as intf
from cvxpy.expressions.operators import BinaryOperator

class Constraint(BinaryOperator):
    __metaclass__ = abc.ABCMeta
    """
    A constraint on an optimization problem of the form
    affine == affine or affine <= affine.
    Stored internally as affine <=/== 0.
    """

    def __repr__(self):
        return self.name()

    def coefficients(self, interface):
        return (self.lh_exp - self.rh_exp).coefficients(interface)

    # Canonicalize the expression in the constraint and
    # add a new constraint with the expression objective.
    def canonicalize(self):
        obj,constraints = (self.lh_exp - self.rh_exp).canonicalize()
        constraints.append(self.__class__(obj, 0))
        return (None, constraints)

    # Does the constraint satisfy DCP requirements?
    @abc.abstractmethod
    def is_dcp(self):
        return NotImplemented

class EqConstraint(Constraint):
    OP_NAME = "=="
    # Both sides must be affine.
    def is_dcp(self):
        return (self.lh_exp - self.rh_exp).curvature().is_affine()

class LeqConstraint(Constraint):
    OP_NAME = "<="
    # Left hand expression must be convex and right hand must be concave.
    def is_dcp(self):
        return (self.lh_exp - self.rh_exp).curvature().is_convex()