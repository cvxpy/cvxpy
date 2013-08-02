from atom import Atom
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.shape import Shape
from monotonicity import Monotonicity
from cvxpy.constraints.second_order import SOC

class norm2(Atom):
    """ L2 norm (sum(x^2))^(1/2) """
    def __init__(self, x):
        self._shape = Shape(1,1)
        super(norm2, self).__init__(x)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Verify that the argument x is a vector.
    def validate_arguments(self):
        rows,cols = self.args[0].size
        if cols != 1:
            raise Exception("The argument '%s' to norm2 must resolve to a vector." 
                % self.args[0].name())

    @staticmethod
    def graph_implementation(var_args):
        x = var_args[0]
        t = Variable()
        return (t, [SOC(t, x)])