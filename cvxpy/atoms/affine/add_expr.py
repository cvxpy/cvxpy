from affine_atom import AffAtom
from ...utilities import coefficient_utils as cu
from ...expressions.constants import Constant
from ... import expressions as exp
from ... import interface as intf
import operator as op

class AddExpression(AffAtom):
    """The sum of any number of expressions.
    """

    def __init__(self, *args):
        super(AddExpression, self).__init__(*args)
        # Promote the terms if necessary.
        rows, cols = self.size
        promoted_args = []
        for arg in self.args:
            promoted_args.append( self._promote(arg) )
        self.args = promoted_args
        self.subexpressions = self.args

    def _promote(self, expr):
        """Promote a scalar expression to a matrix.

        Args:
            expr: The expression to promote.
            rows: The number of rows in the promoted matrix.
            cols: The number of columns in the promoted matrix.

        Returns:
            An expression with size (rows, cols).
        """
        if expr.size == (1,1) and expr.size != self.size:
            ones = Constant(intf.DEFAULT_INTERFACE.ones(*self.size))
            return ones*expr
        else:
            return expr

    def name(self):
        result = str(self.args[0])
        for i in xrange(1, len(self.args)):
            result += " + " + str(self.args[i])
        return result

    def numeric(self, values):
        return reduce(op.add, values)

    # Returns the sign, curvature, and shape.
    def init_dcp_attr(self):
        dcp_attrs = [arg._dcp_attr for arg in self.args]
        self._dcp_attr = reduce(op.add, dcp_attrs)

    # Validate the dimensions.
    def validate_arguments(self):
        shapes = [arg.shape for arg in self.args]
        reduce(op.add, shapes)

    def coefficients(self):
        """Return the dict of Variable to coefficient for the sum.
        """
        coeff_list = [arg.coefficients() for arg in self.args]
        return reduce(cu.add, coeff_list)

    def __add__(self, other):
        """Multiple additions become a single expression rather than a tree.
        """
        terms = self.args[:]
        terms.append(other)
        return AddExpression(*terms)
