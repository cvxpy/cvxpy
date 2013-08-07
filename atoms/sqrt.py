from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import cvxpy.interface.matrix_utilities as intf
import geo_mean as gm

class sqrt(Atom):
    """ Elementwise square root """
    def __init__(self, x):
        super(sqrt, self).__init__(x)

    # The shape is the same as the argument's shape.
    def set_shape(self):
        self._shape = Shape(*self.args[0].size)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONCAVE

    def monotonicity(self):
        return [Monotonicity.INCREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        pass

    def graph_implementation(self, var_args):
        x = var_args[0]
        rows,cols = x.size
        t = Variable(rows, cols)
        constraints = []
        for i in range(rows):
            for j in range(cols):
                obj,constr = gm.geo_mean(x[i,j],1).canonicalize()
                constraints += constr + [AffEqConstraint(obj, t[i,j])]
        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        return sqrt(self.args[0][key])