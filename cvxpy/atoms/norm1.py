from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.sign import Sign
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import abs

class norm1(Atom):
    """ L1 norm sum(|x|) """
    def __init__(self, x):
        super(norm1, self).__init__(x)

    def set_shape(self):
        self._shape = Shape(1,1)

    @property
    def sign(self):
        return Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise Exception("The argument '%s' to norm1 must resolve to a vector." 
                % self.args[0].name())

    def graph_implementation(self, var_args):
        x = var_args[0]
        return sum(abs.abs(x)).canonicalize()