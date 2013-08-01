import cvxpy.settings as s
import cvxpy.interface.matrix_utilities as intf
import expression
from curvature import Curvature
import leaf

class Variable(leaf.Leaf, expression.Expression):
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

    @property
    def size(self):
        return (self.rows, self.cols)

    @property
    def curvature(self):
        return Curvature.AFFINE

    # Save the value of the primal variable.
    def save_value(self, value):
        self.primal_value = value

    @property
    def value(self):
        return self.primal_value

    # Create a new variable that acts as a view into this variable.
    def __getitem__(self, key):
        pass

# class IndexExpression(Expression):
#     # key - a tuple of integers.
#     def __init__(self, expr, key):
#         self.expr = expr
#         self.key = key

#     def name(self):
#         return "%s[%s,%s]" % (self.expr.name(), self.key[0], self.key[1])

#     # TODO slices
#     @property
#     def size(self):
#         return (1,1)

#     # Raise an Exception if the key is not a valid index.
#     def validate_key(self):
#         rows,cols = self.size
#         if not (0 <= self.key[0] and self.key[0] < rows and \
#                 0 <= self.key[1] and self.key[1] < cols): 
#            raise Exception("Invalid indices %s,%s for '%s'." % 
#                 (self.key[0], self.key[1], self.expr.name()))