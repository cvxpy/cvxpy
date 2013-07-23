import cvxpy.interface.matrices as intf
from atom import Atom
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from monotonicity import Monotonicity
from cvxpy.constraints.second_order import SOC

class norm2(Atom):
    """ L2 norm (sum(x^2))^(1/2) """
    def __init__(self, x):
        super(norm2, self).__init__(x)
        self.x = self.args[0]

    def size(self):
        return (1,1)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Represent norm2 as a linear objective and linear and SOC constraints.
    # Verify that the argument x is a vector.
    def base_canonicalize(self):
        rows,cols = self.x.size()
        if cols != 1:
            raise Exception("The argument '%s' to norm2 must resolve to a vector." 
                % self.x.name())
        t = Variable()
        return (t, [ SOC(t,self.x) ])