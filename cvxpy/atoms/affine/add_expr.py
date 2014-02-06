"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from affine_atom import AffAtom
from ...utilities import coefficient_utils as cu
from ...expressions.expression import Expression
from ...expressions.constants import Constant
from ... import expressions as exp
from ... import interface as intf
import operator as op

class AddExpression(AffAtom):
    """The sum of any number of expressions.

    Attributes
    ----------
    prev_sum : AddExpression
        The AddExpression for the first n-1 terms of the sum.

    """

    def __init__(self, terms, prev_sum=None):
        self.prev_sum = prev_sum
        self.args = terms
        self.validate_arguments()
        self.init_dcp_attr()
        self.subexpressions = self.args

    def name(self):
        result = str(self.args[0])
        for i in xrange(1, len(self.args)):
            result += " + " + str(self.args[i])
        return result

    def numeric(self, values):
        return reduce(op.add, values)

    # Validate the dimensions.
    def validate_arguments(self):
        if self.prev_sum is None:
            shapes = (arg.shape for arg in self.args)
            reduce(op.add, shapes)
        else:
            self.prev_sum.shape + self.args[-1].shape

    # Returns the sign, curvature, and shape.
    def init_dcp_attr(self):
        if self.prev_sum is None:
            dcp = (arg._dcp_attr for arg in self.args)
            self._dcp_attr = reduce(op.add, dcp)
        else:
            self._dcp_attr = self.prev_sum._dcp_attr + self.args[-1]._dcp_attr

    def graph_implementation(self, arg_objs):
        return (AddExpression(arg_objs), [])

    def _promote(self, expr):
        """Promote a scalar expression to a matrix.

        Parameters
        ----------
        expr : Expression
            The expression to promote.
        rows : int
            The number of rows in the promoted matrix.
        cols : int
            The number of columns in the promoted matrix.

        Returns
        -------
        Expression
            An expression with size (rows, cols).

        """
        if expr.size == (1, 1) and expr.size != self.size:
            ones = Constant(intf.DEFAULT_INTERFACE.ones(*self.size))
            return ones*expr
        else:
            return expr

    def _tree_to_coeffs(self):
        """Return the dict of Variable to coefficient for the sum.
        """
        # Promote the terms if necessary.
        rows, cols = self.size
        promoted_args = (self._promote(arg) for arg in self.args)
        coeffs = (arg.coefficients() for arg in promoted_args)
        return reduce(cu.add, coeffs)

    def __add__(self, other):
        """Multiple additions become a single expression rather than a tree.
        """
        other = AddExpression.cast_to_const(other)
        return AddExpression(self.args + [other], prev_sum=self)
