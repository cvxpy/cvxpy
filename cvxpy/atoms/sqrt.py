from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
from geo_mean import geo_mean

class sqrt(Atom):
    """ Elementwise square root """
    def __init__(self, x):
        super(sqrt, self).__init__(x)
        # Args are all indexes into x.
        self.x = self.args[0]
        self.args = [xi for xi in self.x]

    # The shape is the same as the argument's shape.
    def set_shape(self):
        self._shape = u.Shape(*self.args[0].size)

    @property
    def sign(self):
        return u.Sign.POSITIVE

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONCAVE

    def monotonicity(self):
        return [u.Monotonicity.INCREASING]
    
    @staticmethod
    def graph_implementation(var_args, size):
        t = Variable(*size)
        constraints = []
        one,dummy = types.constant()(1).canonical_form()
        for ti,xi in zip(t,var_args):
            obj,constr = geo_mean.graph_implementation([xi,one],(1,1))
            constraints += constr + [AffEqConstraint(obj, ti)]
        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        return sqrt(self.x[key])