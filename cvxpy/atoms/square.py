from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import quad_over_lin as qln

class square(Atom):
    """ Elementwise square """
    def __init__(self, x):
        super(square, self).__init__(x)

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
        
    def graph_implementation(self, var_args):
        x = var_args[0]
        rows,cols = x.size
        t = Variable(rows, cols)
        constraints = []
        for i in range(rows):
            for j in range(cols):
                obj,constr = qln.quad_over_lin(x[i,j],1).canonicalize()
                constraints += constr + [AffEqConstraint(obj, t[i,j])]
        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        return square(self.args[0][key])