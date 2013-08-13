import cvxpy.interface.matrix_utilities as intf
from constraint import Constraint
from cvxpy.expressions.operators import BinaryOperator
from affine import AffEqConstraint, AffLeqConstraint
import cvxpy.expressions.types as types

class LeqConstraint(BinaryOperator, Constraint):
    OP_NAME = "<="
    # lh_exp - the left hand side of the constraint.
    # rh_exp - the right hand side of the constraint.
    # value_matrix - the matrix class for storing the dual value.
    # parent - the constraint that produced this constraint as part
    #          of canonicalization.
    def __init__(self, lh_exp, rh_exp, value_matrix=intf.DENSE_TARGET):
        self.value_matrix = value_matrix
        self.interface = intf.get_matrix_interface(self.value_matrix)
        super(LeqConstraint, self).__init__(lh_exp, rh_exp)

    def __repr__(self):
        return self.name()

    @property
    def size(self):
        return self._expr.size

    # The value of the dual variable.
    @property
    def dual(self):
        return self.dual_value

    # Left hand expression must be convex and right hand must be concave.
    def is_dcp(self):
        return self._expr.curvature.is_convex()

    # Replace inequality with an equality with slack.
    def canonicalize(self):
        self._expr = (self.lh_exp - self.rh_exp)
        obj,constr = self._expr.canonical_form()
        dual_holder = AffLeqConstraint(obj, 0, self.value_matrix, self)
        return (None, [dual_holder] + constr)