from max import max
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

class min(max):
    """ Elementwise minimum. """
    @property
    def sign(self):
        return u.Sign.UNKNOWN
        
    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONCAVE

    def graph_implementation(self, var_args):
        t = Variable(*self.size)
        constraints = [AffLeqConstraint(t, x) for x in var_args]
        return (t, constraints)