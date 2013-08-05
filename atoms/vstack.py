from atom import Atom
import cvxpy.expressions.types as types
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.curvature import Curvature
from cvxpy.expressions.shape import Shape
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint
from monotonicity import Monotonicity
import cvxpy.interface.matrix_utilities as intf

class vstack(Atom):
    """ Vertical concatenation """
    def __init__(self, *args)):
        super(vstack, self).__init__(*args)

    # The shape is the common width and the sum of the heights.
    def set_shape(self):
        cols = self.args[0].size[1]
        rows = sum(arg.size[0] for arg in self.args)
        self._shape = Shape(rows, cols)

    # Default curvature.
    def base_curvature(self):
        return Curvature.AFFINE

    def monotonicity(self): # TODO what would make sense?
        return len(self.args)*[Monotonicity.INCREASING]

    # Any argument size is valid.
    def validate_arguments(self):
        arg_cols = [arg.size[1] for arg in self.args]
        if max(arg_cols) != min(arg_cols):
            raise Exception("All arguments to vstack must have the same width.")

    def graph_implementation(self, var_args):
        t = Variable(*self.size)
        constraints = []
        offset = 0
        for arg in var_args:
            rows,cols = arg.size
            for i in range(rows):
                for j in range(cols):
                    constraints.append( AffEqConstraint(t[i+offset,j], arg[i,j]) )
            offset += rows

        return (t, constraints)

    # Return the absolute value of the argument at the given index.
    def index_object(self, key):
        index = 0
        offset = self.args[0].size[0]
        while offset < key[0]:
            index += 1
            offset += self.args[index].size[0]
        return self.args[index][key]