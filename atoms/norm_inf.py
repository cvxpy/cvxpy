import cvxpy.interface.matrices as intf
from atom import Atom
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from monotonicity import Monotonicity

class normInf(Atom):
    """ Infinity norm max{|x|} """
    def __init__(self, x):
        super(normInf, self).__init__(x)
        self.x = self.args[0]

    def size(self):
        return (1,1)

    # Default curvature.
    def base_curvature(self):
        return Curvature.CONVEX

    def monotonicity(self):
        return [Monotonicity.NONMONOTONIC]

    # Represent normInf as a linear objective and linear constraints.
    # Verify that the argument x is a vector.
    def base_canonicalize(self):
        rows,cols = self.x.size()
        if cols != 1: #TODO put validation into Atom
            raise Exception("The argument '%s' to normInf must resolve to a vector." 
                % self.x.name())
        t = Variable()
        ones = intf.ones(rows, cols)
        return (t, [-ones*t <= self.x, self.x <= ones*t])