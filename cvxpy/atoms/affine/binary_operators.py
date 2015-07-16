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

from __future__ import division
import sys

from cvxpy.atoms.affine.affine_atom import AffAtom
import cvxpy.interface as intf
from cvxpy.expressions.constants import Constant
import cvxpy.lin_ops.lin_utils as lu
import operator as op
import numpy as np
if sys.version_info >= (3, 0):
    from functools import reduce

class BinaryOperator(AffAtom):
    """
    Base class for expressions involving binary operators.

    """
    def __init__(self, lh_exp, rh_exp):
        super(BinaryOperator, self).__init__(lh_exp, rh_exp)

    def name(self):
        return ' '.join([str(self.args[0].name()),
                         self.OP_NAME,
                         str(self.args[1].name())])

    # Applies the binary operator to the values.
    def numeric(self, values):
        return reduce(self.OP_FUNC, values)

    # Sets the sign, curvature, and shape.
    def init_dcp_attr(self):
        self._dcp_attr = self.OP_FUNC(self.args[0]._dcp_attr,
                                      self.args[1]._dcp_attr)

    # Validate the dimensions.
    def validate_arguments(self):
        self.OP_FUNC(self.args[0]._dcp_attr.shape,
                     self.args[1]._dcp_attr.shape)

class MulExpression(BinaryOperator):
    OP_NAME = "*"
    OP_FUNC = op.mul

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Multiply the linear expressions.

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
        # Promote the right hand side to a diagonal matrix if necessary.
        if size[1] != 1 and arg_objs[1].size == (1, 1):
            arg = lu.promote(arg_objs[1], (size[1], 1))
            arg_objs[1] = lu.diag_vec(arg)
        return (lu.mul_expr(arg_objs[0], arg_objs[1], size), [])

class RMulExpression(MulExpression):
    """Multiplication by a constant on the right.
    """

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Multiply the linear expressions.

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
        # Promote the left hand side to a diagonal matrix if necessary.
        if size[0] != 1 and arg_objs[0].size == (1, 1):
            arg = lu.promote(arg_objs[0], (size[0], 1))
            arg_objs[0] = lu.diag_vec(arg)
        return (lu.rmul_expr(arg_objs[0], arg_objs[1], size), [])

class DivExpression(BinaryOperator):
    OP_NAME = "/"
    OP_FUNC = op.__truediv__ if (sys.version_info >= (3,0) ) else op.__div__

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Multiply the linear expressions.

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
        return (lu.div_expr(arg_objs[0], arg_objs[1]), [])
