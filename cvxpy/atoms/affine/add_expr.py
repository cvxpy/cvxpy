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

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants import Constant
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import operator as op

class AddExpression(AffAtom):
    """The sum of any number of expressions.
    """

    def __init__(self, terms):
        # TODO call super class init?
        self._dcp_attr = reduce(op.add, [t._dcp_attr for t in terms])
        self.args = []
        for term in terms:
            self.args += self.expand_args(term)
        self.subexpressions = self.args

    def expand_args(self, expr):
        """Helper function to extract the arguments from an AddExpression.
        """
        if isinstance(expr, AddExpression):
            return expr.args
        else:
            return [expr]

    def name(self):
        result = str(self.args[0])
        for i in xrange(1, len(self.args)):
            result += " + " + str(self.args[i])
        return result

    def numeric(self, values):
        return reduce(op.add, values)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Sum the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        for i, arg in enumerate(arg_objs):
            if arg.size != size:
                arg_objs[i] = lu.promote(arg, size)
        return (lu.sum_expr(arg_objs), [])
