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
import cvxpy.lin_ops.lin_utils as lu
import operator as op


class UnaryOperator(AffAtom):
    """
    Base class for expressions involving unary operators.
    """

    def __init__(self, expr):
        super(UnaryOperator, self).__init__(expr)

    def name(self):
        return self.OP_NAME + self.args[0].name()

    # Applies the unary operator to the value.
    def numeric(self, values):
        return self.OP_FUNC(values[0])


class NegExpression(UnaryOperator):
    OP_NAME = "-"
    OP_FUNC = op.neg

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return self.args[0].size

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return (self.args[0].is_negative(), self.args[0].is_positive())

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return True

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Negate the affine objective.

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
        return (lu.neg_expr(arg_objs[0]), [])
