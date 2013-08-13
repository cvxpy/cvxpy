from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
from quad_over_lin import quad_over_lin

class square(Atom):
    """ Elementwise square """
    def __init__(self, x):
        super(square, self).__init__(x)
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
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.Monotonicity.NONMONOTONIC]
    
    @staticmethod
    def graph_implementation(var_args, size):
        t = Variable(*size)
        constraints = []
        one,dummy = types.constant()(1).canonical_form()
        for ti,xi in zip(t,var_args):
            obj,constr = quad_over_lin.graph_implementation([xi,one],(1,1))
            constraints += constr + [AffEqConstraint(obj, ti)]
        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        return square(self.x[key])