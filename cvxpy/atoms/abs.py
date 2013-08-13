from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

class abs(Atom):
    """ Elementwise absolute value """
    def __init__(self, x):
        super(abs, self).__init__(x)

    # The shape is the same as the argument's shape.
    def set_shape(self):
        self._shape = u.Shape(*self.args[0].size)

    @property
    def sign(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.Monotonicity.NONMONOTONIC]
    
    @staticmethod
    def graph_implementation(var_args, size):
        x = var_args[0]
        t = Variable(*size)
        constraints = [AffLeqConstraint(-t, x), 
                       AffLeqConstraint(x, t)]
        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        return abs(self.args[0][key])