import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
import expression
from curvature import Curvature
import leaf

class Variable(object):
    """ A dummy type to mark and create variables. """
    # Return a scalar or matrix variable depending on the dimensions.
    def __new__(cls, rows=1, cols=1, name=None, value_matrix=intf.DENSE_TARGET):
        if (rows,cols) == (1,1):
            return BaseVariable.__new__(ScalarVariable)
        else:
            return BaseVariable.__new__(MatrixVariable)

class BaseVariable(leaf.Leaf, expression.Expression):
    """ The base variable class """
    VAR_COUNT = 0        
    # name - unique identifier.
    # rows - variable height.
    # cols - variable width.
    # value_matrix - the matrix type used to store values.
    def __init__(self, rows=1, cols=1, name=None, value_matrix=intf.DENSE_TARGET):
        self._rows = rows
        self._cols = cols
        self._init_id()
        self._name = self.id if name is None else name
        self.interface = intf.get_matrix_interface(value_matrix)
        self.primal_value = None

    # Ensure Variable __new__ is only called once.
    def __new__(cls, *args, **kwargs):
        return leaf.Leaf.__new__(cls, *args, **kwargs)

    # Initialize the id.
    def _init_id(self):
        self.id = BaseVariable.next_var_name()

    # Returns a new variable name based on a global counter.
    @staticmethod
    def next_var_name():
        BaseVariable.VAR_COUNT += 1
        return "%s%d" % (s.VAR_PREFIX, BaseVariable.VAR_COUNT)

    def name(self):
        return self._name

    @property
    def size(self):
        return (self._rows, self._cols)

    @property
    def curvature(self):
        return Curvature.AFFINE

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

class MatrixVariable(BaseVariable, Variable):
    """ A matrix variable """
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

    # Raise an Exception if the key is not a valid index.
    def validate_key(self, key):
        rows,cols = self.size
        if not (0 <= key[0] and key[0] < rows and \
                0 <= key[1] and key[1] < cols): 
           raise Exception("Invalid indices %s,%s for '%s'." % 
                (key[0], key[1], self.name()))

    # Create a new variable that acts as a view into this variable.
    # Updating the variable's value updates the value of this variable instead.
    def __getitem__(self, key):
        self.validate_key(key)
        return IndexVariable(self, key)
        # TODO # Set value if variable has value.
        # if self.value is not None:
        #     index_var.primal_value = self.value[key]

    # Iterating over the variable returns the index variables
    # in column major order.
    def __iter__(self):
        for row in range(self.size[0]):
            for col in range(self.size[1]):
                yield self[row,col]

    # The id of the view at the given index.
    def index_id(self, row, col):
        return "%s[%s,%s]" % (self.id, row, col)

    # # Recover the value matrix from the index variables.
    # def recover_value_matrix(self, key, value):
    #     if self.primal_value is None:
    #         self.primal_value = self.interface.zeros(self.size[0], self.size[1])
    #     self.primal_value[key[0],key[1]] = value

class ScalarVariable(BaseVariable, Variable):
    """ A scalar variable """
    # Initialized with identity matrix as variable's coefficient.
    def coefficients(self, interface):
        return {self.id: 1}

    # The id of the view at the given index.
    def index_id(self, row, col):
        return self.id

class IndexVariable(ScalarVariable):
    """ An index into a matrix variable """
    # parent - the variable indexed into.
    # key - the index (row,col).
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        name = "%s[%s,%s]" % (parent.name(), key[0], key[1])
        super(IndexVariable, self).__init__(name=name)

    # Initialize the id.
    def _init_id(self):
        self.id = self.parent.index_id(*self.key)

    # Return parent so that the parent value is updated.
    def terms(self):
        return [self.parent]