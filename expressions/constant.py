import expression
import leaf
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from curvature import Curvature
from shape import Shape

class Constant(leaf.Leaf):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value, name=None):
        self.value = value
        self.param_name = name
        self.set_shape()
        super(Constant, self).__init__()

    def name(self):
        return str(self.value) if self.param_name is None else self.param_name

    @property
    def curvature(self):
        return Curvature.CONSTANT

    # The constant's shape is fixed.
    def set_shape(self):
        self._shape = Shape(*intf.size(self.value))

    # Return the constant value, converted to the target matrix.
    def coefficients(self, interface):
        return {s.CONSTANT: interface.const_to_matrix(self.value)}