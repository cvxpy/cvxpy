import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
import expression
from curvature import Curvature
from shape import Shape
import leaf
from collections import deque

class Variable(leaf.Leaf):
    """ The base variable class """
    VAR_COUNT = 0        
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    # value_matrix - the matrix type used to store values.
    def __init__(self, rows=1, cols=1, name=None, value_matrix=intf.DENSE_TARGET):
        self._shape = Shape(rows, cols)
        self._init_id()
        self._name = self.id if name is None else name
        self.interface = intf.get_matrix_interface(value_matrix)
        self.primal_value = None
        super(Variable, self).__init__()

    # Initialize the id.
    def _init_id(self):
        self.id = Variable.next_var_name()

    # Returns a new variable name based on a global counter.
    @staticmethod
    def next_var_name():
        Variable.VAR_COUNT += 1
        return "%s%d" % (s.VAR_PREFIX, Variable.VAR_COUNT)

    def name(self):
        return self._name

    @property
    def curvature(self):
        return Curvature.AFFINE

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    # Returns the id's of the index variables, each with a matrix
    # of the same dimensions as the variable that is 0 except
    # for at the index key, where it is 1.
    def coefficients(self, interface):
        coeffs = {}
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                id = self.index_id(row, col)
                matrix = interface.zeros(*self.size)
                matrix[row,col] = 1
                coeffs[id] = matrix
        return coeffs

    # The id of the view at the given index.
    def index_id(self, row, col):
        return "%s[%s,%s]" % (self.id, row, col)

    # Return a scalar view into a matrix variable.
    def index_object(self, key):
        return IndexVariable(self, key)

class IndexVariable(Variable):
    """ An index into a matrix variable """
    # parent - the variable indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        name = "%s[%s,%s]" % (parent.name(), key[0], key[1])
        super(IndexVariable, self).__init__(name=name)

    # Coefficient for a scalar variable.
    def coefficients(self, interface):
        return {self.id: 1}

    # Initialize the id.
    def _init_id(self):
        self.id = self.parent.index_id(*self.key)

    # Return parent so that the parent value is updated.
    def as_term(self):
        return (self.parent, deque([self]))

    # The value at the index.
    @property
    def value(self):
        return self.parent.value[self.key]