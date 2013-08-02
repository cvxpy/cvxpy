from cvxpy.expressions.variable import Variable
import cvxpy.interface.matrix_utilities as intf
from affine import AffEqConstraint, AffLeqConstraint

class SOC(object):
    """ 
    A second-order cone constraint:
        norm2(x) <= t
    """
    def __init__(self, t, x):
        self.x = x
        self.t = t
        super(SOC, self).__init__()

    # Reduce SOC to affine/basic SOC constraints and 
    # a SOC with variables as arguments (i.e. basic).
    def canonicalize(self):
        x_obj,x_constraints = self.x.canonical_form()
        t_obj,t_constraints = self.t.canonical_form()
        constraints = x_constraints + t_constraints

        vector = Variable(self.x.size[0])
        constraints.append( AffEqConstraint(vector, x_obj) )

        scalar = Variable()
        constraints.append( AffEqConstraint(scalar, t_obj) )
        return (None, constraints + [SOC(scalar,vector)])

    # Formats SOC constraints for the solver.
    def format(self):
        return [AffLeqConstraint(-self.t, 0), 
                AffLeqConstraint(-self.x, 0)]

    # The dimensions of the second-order cone.
    @property
    def size(self):
        return self.x.size[0] + self.t.size[0]