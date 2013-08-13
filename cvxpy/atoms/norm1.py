from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
from abs import abs

class norm1(Atom):
    """ L1 norm sum(|x|) """
    def __init__(self, x):
        super(norm1, self).__init__(x)

    def set_shape(self):
        self.validate_arguments()
        self._shape = u.Shape(1,1)

    @property
    def sign(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.Monotonicity.NONMONOTONIC]

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise Exception("The argument '%s' to norm1 must resolve to a vector." 
                % self.args[0].name())
    
    @staticmethod
    def graph_implementation(var_args, size):
        x = var_args[0]
        obj,constraints = abs.graph_implementation([x], x.size)
        return (sum(obj),constraints)