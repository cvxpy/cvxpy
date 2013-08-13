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
        expr_obj,expr_constr = self._expr.canonical_form()
        dual_holder = AffEqConstraint(expr_obj, 0, self.value_matrix, self)
        return (None, [dual_holder] + expr_constr)