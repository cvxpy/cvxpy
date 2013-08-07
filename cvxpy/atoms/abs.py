from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import cvxpy.interface.matrix_utilities as intf

class abs(Atom):
    """ Elementwise absolute value """
    def __init__(self, x):
        super(abs, self).__init__(x)

    # The shape is the same as the argument's shape.
    def set_shape(self):
        self._shape = Shape(*self.args[0].size)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Any argument size is valid.
    def validate_arguments(self):
        pass

    def graph_implementation(self, var_args):
        x = var_args[0]
        t = Variable(*x.size)
        constraints = [AffLeqConstraint(-t, x), 
                       AffLeqConstraint(x, t)]
        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        return abs(self.args[0][key])