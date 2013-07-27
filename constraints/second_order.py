from cvxpy.expressions.variable import Variable
import cvxpy.interface.matrix_utilities as intf

class SOC(object):
    """ 
    A second-order cone constraint:
        norm2(x) <= t
    """
    def __init__(self, t, x):
        self.x = x
        self.t = t

    # Reduce SOC to affine/basic SOC constraints and 
    # a SOC with variables as arguments (i.e. basic).
    def canonicalize(self):
        x_obj,x_constraints = self.x.canonicalize()
        t_obj,t_constraints = self.t.canonicalize()
        constraints = x_constraints + t_constraints

        vector = Variable(self.x.size[0])
        constraints.append(vector == x_obj)

        scalar = Variable()
        constraints.append(scalar == t_obj)
        return (None, constraints + [SOC(scalar,vector)])

    # Formats SOC constraints for the solver.
    def format(self):
        return [-self.t <= 0, -self.x <= 0]

    # The dimensions of the second-order cone.
    @property
    def size(self):
        return self.x.size[0] + self.t.size[0]