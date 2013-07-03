import settings
from coefficients import Coeff
from shape import Shape
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
        return "%s%d" % (settings.VAR_PREFIX, Variable.VAR_COUNT)

    def name(self):
        return self.var_name

    def coefficients(self):
        return Coeff({self.name():1})

    def shape(self):
        return Shape(self.rows,1)

    def variables(self):
        return {self.name(): self}