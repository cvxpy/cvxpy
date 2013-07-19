import cvxpy.settings as s
import cvxpy.interface.matrices as intf
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
        self.var_name = Variable.next_var_name() if name is None else name

    # Returns a new variable name based on a global counter.
    @staticmethod
    def next_var_name():
        Variable.VAR_COUNT += 1
        return "%s%d" % (s.VAR_PREFIX, Variable.VAR_COUNT)

    def name(self):
        return self.var_name

    # Initialized with identity matrix as variable's coefficient.
    def coefficients(self):
        return {self.name(): intf.identity(self.rows)}

    def variables(self):
        return {self.name(): self}

    def size(self):
        return (self.rows, self.cols)

    def curvature(self):
        return Curvature.AFFINE

    def canonicalize(self):
        return (self,[])