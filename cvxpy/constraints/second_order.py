from cvxpy.expressions.variable import Variable
import cvxpy.interface.matrix_utilities as intf
from affine import AffEqConstraint, AffLeqConstraint

class SOC(object):
    """ 
    A second-order cone constraint:
        norm2(x) <= t
    """
    # x - an affine expression or objective.
    # t - an affine expression or objective.
    def __init__(self, t, x):
        self.x = x
        self.t = t
        super(SOC, self).__init__()

    # Formats SOC constraints for the solver.
    def format(self):
        return [AffLeqConstraint(-self.t, 0), 
                AffLeqConstraint(-self.x, 0)]

    # The dimensions of the second-order cone.
    @property
    def size(self):
        return self.x.size[0] + self.t.size[0]