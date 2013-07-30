import cvxpy.settings as s
import expression
import leaf
from curvature import Curvature

class Parameter(leaf.Leaf, expression.Expression):
    """
    A parameter, either matrix or scalar.
    """
    def __init__(self, rows, cols, name):
        self.rows = rows
        self.cols = cols
        self.param_name = name

    def name(self):
        return self.param_name

    # Return the Parameter itself as the coefficient value.
    def coefficient(self, interface):
        return self

    # Part of the constant term's coefficients.
    @property
    def id(self):
        return s.CONSTANT

    @property
    def size(self):
        return (self.rows, self.cols)

    @property
    def curvature(self):
        return Curvature.CONSTANT