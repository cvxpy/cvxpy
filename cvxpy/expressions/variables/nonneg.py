from cvxpy.expressions.variables.variable import Variable
import cvxpy.lin_ops.lin_utils as lin_utils
import cvxpy.utilities as utils

class NonNegative(Variable):

    # Rewrite the variable so that it's in epigraph form; the constraint below simply enforces that t >= 0
    def canonicalize(self):
        t = lin_utils.create_var(self.size, self.id)
        return (t, [lin_utils.create_geq(t)])

    def __repr__(self):
        return "NonNegative(%d, %d)" % self.size

    # Override
    def init_dcp_attr(self):
        self._dcp_attr = utils.DCPAttr(utils.Sign.POSITIVE,
                                       utils.Curvature.AFFINE,
                                       utils.Shape(self._rows, self._cols))