from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.sign import Sign
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import cvxpy.interface.matrix_utilities as intf

class max(Atom):
    """ Maximum element in all arguments. """
    # The shape is the same as the argument's shape.
    def set_shape(self):
        self._shape = Shape(1,1)

    @property
    def sign(self):
        return Sign.UNKNOWN

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return len(self.args)*[Monotonicity.INCREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        pass

    def graph_implementation(self, var_args):
        t = Variable()
        constraints = [AffLeqConstraint(x, t) for x in var_args]
        return (t, constraints)