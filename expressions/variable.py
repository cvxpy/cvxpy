import settings as s
import interface.matrices as intf
from expression import Expression

class Variable(Expression):
    """
    A vector variable.
    name - unique identifier.
    rows - vector dimension.
    """
    VAR_COUNT = 0

    def __init__(self, rows=1,name=None):
        self.rows = rows
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
        mat,sizes = intf.const_to_matrix(1, intf.TARGET_MATRIX)
        mat = intf.conform_to_shapes( mat, set([(self.rows, self.rows)]) ) # TODO shapeset
        return ( {self.name(): val}, set([(self.rows,1)]) )

    def variables(self):
        return {self.name(): self}