from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature

class abs(Expression):
    """ Absolute value |x| """
    def __init__(self, x):
        self.x = x

    def name(self):
        return "abs(%s)" % self.x.name()

    def size(self):
        return (1,1)

    # TODO DCP rules
    def curvature(self):
        if self.x.curvature().is_convex():
            return Curvature.CONVEX
        else:
            return Curvature.UNKNOWN

    def canonicalize(self):
        t = Variable()
        return (t, [-t <= self.x, self.x <= t])
