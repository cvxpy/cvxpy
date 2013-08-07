import constant
from shape import Shape

class Parameter(constant.Constant):
    """
    A parameter, either matrix or scalar.
    """
    def __init__(self, rows=1, cols=1, name=None):
        self._rows = rows
        self._cols = cols
        super(Parameter, self).__init__(None, name)

    # The constant's shape is fixed.
    def set_shape(self):
        self._shape = Shape(self._rows, self._cols)