import expression
import leaf
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from curvature import Curvature

class Constant(leaf.Leaf, expression.Expression):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value, name=None):
        self.value = value
        self.param_name = name

    def name(self):
        return str(self.value) if self.param_name is None else self.param_name

    @property
    def size(self):
        return intf.size(self.value)

    @property
    def curvature(self):
        return Curvature.CONSTANT

    # Return the constant value, converted to the target matrix.
    def coefficient(self, interface):
        return interface.const_to_matrix(self.value)

    # Part of the constant term's coefficients.
    @property
    def id(self):
        return s.CONSTANT