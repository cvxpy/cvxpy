import expression
import leaf
import cvxpy.interface.matrix_utilities as intf
import cvxpy.settings as s
from curvature import Curvature
from sign import Sign
from shape import Shape

class Constant(leaf.Leaf):
    """
    A constant, either matrix or scalar.
    """
    def __init__(self, value, name=None):
        self.value = value
        self.param_name = name
        self.set_shape()
        self.set_sign()
        super(Constant, self).__init__()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def name(self):
        return str(self.value) if self.param_name is None else self.param_name

    @property
    def curvature(self):
        return Curvature.CONSTANT

    # The constant's shape is fixed.
    def set_shape(self):
        self._shape = Shape(*intf.size(self.value))

    # The constant's sign is fixed.
    def set_sign(self):
        if self.size == (1,1):
            self._sign = Sign.val_to_sign(intf.scalar_value(self.value))
        else:
            self._sign = Sign.POSITIVE

    # Return the constant value, converted to the target matrix.
    def coefficients(self, interface):
        return {s.CONSTANT: interface.const_to_matrix(self.value)}

    # Return a scalar view into a matrix constant.
    def index_object(self, key):
        return IndexConstant(self, key)


class IndexConstant(Constant):
    """ An index into a matrix constant """
    # parent - the constant indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        self._shape = Shape(1,1)
        super(IndexConstant, self).__init__(
            intf.index(self.parent.value, self.key)
        )