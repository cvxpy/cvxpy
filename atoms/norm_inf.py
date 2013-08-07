from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import max
import abs

class normInf(Atom):
    """ Infinity norm max{|x|} """
    def __init__(self, x):
        super(normInf, self).__init__(x)

    def set_shape(self):
        self._shape = Shape(1,1)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise TypeError("The argument '%s' to normInf must resolve to a vector." 
                % self.args[0].name())

    def graph_implementation(self, var_args):
        x = var_args[0]
        return max.max(abs.abs(x)).canonicalize()