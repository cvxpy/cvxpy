from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import cvxpy.interface.matrix_utilities as intf
import vstack

class quad_over_lin(Atom):
    """ x'*x/y """
    def __init__(self, x, y):
        super(quad_over_lin, self).__init__(x, y)

    # The shape is the common width and the sum of the heights.
    def set_shape(self):
        self._shape = Shape(1,1)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self): # TODO what would make sense?
        return [Monotonicity.NONMONOTONIC, Monotonicity.DECREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        if not self.args[0].is_vector():
            raise TypeError("The first argument to quad_over_lin must be a vector.")
        elif not self.args[1].is_scalar():
            raise TypeError("The seconde argument to quad_over_lin must be a scalar.")

    def graph_implementation(self, var_args):
        v = Variable(*self.size)
        x = var_args[0]
        y = var_args[1]

        constraints = SOC(y + v, vstack.vstack(y - v, 2*x)).canonicalize()[1]
        constraints += [AffLeqConstraint(0, y)]
        return (v, constraints)