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

class MulExpression(AffAtom):
    """The product of any number of expressions.
    """

    def __init__(self, terms):
        self._dcp_attr = reduce(op.mul, [t._dcp_attr for t in terms])
        self.args = []
        for term in terms:
            self.args += self.expand_args(term)
        self.subexpressions = self.args

    def expand_args(self, expr):
        """Helper function to extract the arguments from an AddExpression.
        """
        if isinstance(expr, MulExpression):
            return expr.args
        else:
            return [expr]

    def name(self):
        result = str(self.args[0])
        for i in xrange(1, len(self.args)):
            result += " * " + str(self.args[i])
        return result

    def numeric(self, values):
        return reduce(op.mul, values)

    def graph_implementation(self, arg_objs):
        return (MulExpression(arg_objs), [])

    def _tree_to_coeffs(self):
        """Return the dict of Variable to coefficient for the sum.
        """
        coeffs = (arg.coefficients() for arg in self.args)
        return reduce(cu.mul, coeffs)
