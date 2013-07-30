from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constant import Constant

class Variables(object):
    """
    Constructs a namespace of variables from [name,rows,cols] lists.
    A string variable will be interpreted as [name,1,1], and
    a list [name,rows] will be interpreted as [name,rows,1].
    """
    def __init__(self, *args):
        for arg in args:
            # Scalar variable.
            if isinstance(arg, str):
                name = arg
                var = Variable(name=arg)
            # Vector variable.
            elif isinstance(arg, list) and len(arg) == 2:
                name = arg[0]
                var = Variable(name=arg[0],rows=arg[1])
            # Matrix variable.
            elif isinstance(arg, list) and len(arg) == 3:
                name = arg[0]
                var = Variable(name=arg[0],rows=arg[1],cols=arg[2])
            else:
                raise Exception("Invalid argument '%s' to 'Variables'." % arg)
            setattr(self, name, var)

class Constants(object):
    """
    Constructs a namespace of constants from [name,value] lists.
    """
    def __init__(self, *args):
        for arg in args:
            if isinstance(arg, list) and len(arg) == 2:
                name = arg[0]
                const = Constant(arg[1], name)
            else:
                raise Exception("Invalid argument '%s' to 'Constants'." % arg)
            setattr(self, name, const)