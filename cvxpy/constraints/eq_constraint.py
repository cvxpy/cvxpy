from leq_constraint import LeqConstraint
from affine import AffEqConstraint, AffLeqConstraint

class EqConstraint(LeqConstraint):
    OP_NAME = "=="
    # Both sides must be affine.
    def is_dcp(self):
        return self._expr.curvature.is_affine()

    # TODO expanding equality constraints.
    # Verify doesn't affect dual variables.
    def canonicalize(self):
        dual_holder = AffEqConstraint(self._expr_obj, 0, self.value_matrix, self)
        return (None, [dual_holder] + self._expr_constr)