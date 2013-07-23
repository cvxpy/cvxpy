from cvxpy.expressions.variable import Variable
import cvxpy.interface.matrices as intf

class SDC(object):
    """ 
    A positive semidefinite cone constraint:
        x'Ax >= 0 for all x
    """
    def __init__(self, A):
        self.A = A

    # Reduce SDC to affine constraints and 
    # a SDC with a single matrix variable as an argument.
    def canonicalize(self):
        x_obj,x_constraints = self.x.canonicalize()
        t_obj,t_constraints = self.t.canonicalize()
        constraints = x_constraints + t_constraints

        vector = Variable(self.x.size()[0])
        constraints.append(vector == x_obj)

        scalar = Variable()
        constraints.append(scalar == t_obj)
        return (None, constraints + [SDC(scalar,vector)])

    # Formats SDC constraints for the solver.
    def format(self):
        return [-self.t <= 0,
                -self.x <= intf.zeros(self.size()-1,1)]

    # The dimensions of the second-order cone.
    def size(self):
        return self.x.size()[0] + self.t.size()[0]