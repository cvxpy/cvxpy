from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from monotonicity import Monotonicity

class normInf(Atom):
    """ Infinity norm max{|x|} """
    def __init__(self, x):
        super(normInf, self).__init__(x)
        self.x = self.args[0]

    @property
    def size(self):
        return (1,1)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        rows,cols = self.x.size
        if cols != 1:
            raise Exception("The argument '%s' to normInf must resolve to a vector." 
                % self.x.name())

    @staticmethod
    def graph_implementation(var_args):
        x = var_args[0]
        rows,cols = x.size
        t = Variable()
        ones = types.constant()(rows*[1])
        return (t, [-ones*t <= x, x <= ones*t])