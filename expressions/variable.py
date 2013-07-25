import inspect
import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
from expression import Expression
from curvature import Curvature

class Variable(Expression):
    """
    A matrix variable.
    """
    VAR_COUNT = 0
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    def __init__(self, rows=1, cols=1, name=None):
        self.rows = rows
        self.cols = cols
        self.id = Variable.next_var_name()
        self.var_name = self.id if name is None else name

    # Returns a new variable name based on a global counter.
    @staticmethod
    def next_var_name():
        Variable.VAR_COUNT += 1
        return "%s%d" % (s.VAR_PREFIX, Variable.VAR_COUNT)

    def name(self):
        return self.var_name

    # Initialized with identity matrix as variable's coefficient.
    def coefficients(self, interface):
        return {self.id: interface.identity(self.rows)}

    def variables(self):
        return {self.id: self}

    def size(self):
        return (self.rows, self.cols)

    def curvature(self):
        return Curvature.AFFINE

    def canonicalize(self):
        return (self,[])

class Variables(object):
    """
    Constructs a dictionary of variables from (name,rows,cols) tuples.
    A string variable will be interpreted as (name,1,1), and
    a tuple (name,rows) will be interpreted as (name,rows,1).
    """
    def __init__(self, *args):
        for arg in args:
            # Scalar variable.
            if isinstance(arg, str):
                name = arg
                var = Variable(name=arg)
            # Vector variable.
            elif isinstance(arg, tuple) and len(arg) == 2:
                name = arg[0]
                var = Variable(name=arg[0],rows=arg[1])
            # Matrix variable.
            elif isinstance(arg, tuple) and len(arg) == 3:
                name = arg[0]
                var = Variable(name=arg[0],rows=arg[1],cols=arg[2])
            else:
                raise Exception("Invalid argument '%s' to 'variables'." % arg)
            setattr(self, name, var)