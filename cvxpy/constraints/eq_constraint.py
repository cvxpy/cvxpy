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
        self._expr = (self.lh_exp - self.rh_exp)
        obj,constr = self._expr.canonical_form()
        dual_holder = AffEqConstraint(obj, 0, self.value_matrix, self)
        return (None, [dual_holder] + constr)