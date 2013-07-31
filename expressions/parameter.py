import constant

class Parameter(constant.Constant):
    """
    A parameter, either matrix or scalar.
    """
    def __init__(self, name=None):
        super(Parameter, self).__init__(None, name)