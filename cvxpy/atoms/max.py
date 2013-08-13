from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

class max(Atom):
    """ Elementwise maximum. """
    # The shape is the common shape of all the arguments.
    def set_shape(self):
        shape = self.args[0]._shape
        for arg in self.args[1:]:
            shape = shape + arg._shape
        self._shape = shape

    @property
    def sign(self):
        return u.Sign.UNKNOWN

    # Default curvature.
    def base_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return len(self.args)*[u.Monotonicity.INCREASING]

    def graph_implementation(self, var_args):
        t = Variable(*self.size)
        constraints = [AffLeqConstraint(x, t) for x in var_args]
        return (t, constraints)

    # Return the max of the arguments' elements at the given index.
    def index_object(self, key):
        args = []
        for arg in self.args:
            if arg.size == (1,1):
                args.append(arg)
            else:
                args.append(arg[key])
        return self.__class__(*args)