import cvxpy.interface.matrices as intf
from atom import Atom
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from monotonicity import Monotonicity

class norm1(Atom):
    """ L1 norm sum(|x|) """
    def __init__(self, x):
        super(norm1, self).__init__(x)
        self.x = self.args[0]

    def size(self):
        return (1,1)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Represent norm1 as a linear objective and linear constraints.
    # Verify that the argument x is a vector.
    def base_canonicalize(self):
        rows,cols = self.x.size()
        if cols != 1:
            raise Exception("The argument '%s' to norm1 must resolve to a vector." 
                % self.x.name())
        t = Variable(rows)
        ones = intf.ones(1, rows)
        return (ones*t,[-t <= self.x, self.x <= t])