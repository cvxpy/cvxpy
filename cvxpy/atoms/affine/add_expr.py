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

import sys
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.constants import Constant
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import operator as op
if sys.version_info >= (3, 0):
    from functools import reduce

class AddExpression(AffAtom):
    """The sum of any number of expressions.
    """

    def __init__(self, arg_groups):
        # For efficiency group args as sums.
        self._arg_groups = arg_groups
        super(AddExpression, self).__init__(*arg_groups)
        self.args = []
        for group in arg_groups:
            self.args += self.expand_args(group)

    def init_dcp_attr(self):
        self._dcp_attr = reduce(op.add, [arg._dcp_attr for arg in self.args])

    def expand_args(self, expr):
        """Helper function to extract the arguments from an AddExpression.
        """
        if isinstance(expr, AddExpression):
            return expr.args
        else:
            return [expr]

    def name(self):
        result = str(self.args[0])
        for i in range(1, len(self.args)):
            result += " + " + str(self.args[i])
        return result

    def numeric(self, values):
        return reduce(op.add, values)

    # As __init__ takes in the arg_groups instead of args, we need a special
    # copy() function.
    def copy(self, args=None):
        """Returns a shallow copy of the AddExpression atom.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the atom. If args=None, use the
            current args of the atom.

        Returns
        -------
        AddExpression atom
        """
        if args is None:
            args = self._arg_groups
        # Takes advantage of _arg_groups if present for efficiency.
        copy = type(self).__new__(type(self))
        copy.__init__(args)
        return copy

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
