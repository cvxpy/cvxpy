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
    # value_matrix - the matrix type used to store values.
    def __init__(self, rows=1, cols=1, name=None, value_matrix=intf.DENSE_TARGET):
        self.rows = rows
        self.cols = cols
        self.id = Variable.next_var_name()
        self.var_name = self.id if name is None else name
        self.interface = intf.get_matrix_interface(intf.DENSE_TARGET)

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

    @property
    def size(self):
        return (self.rows, self.cols)

    @property
    def curvature(self):
        return Curvature.AFFINE

    def canonicalize(self):
        return (self,[])