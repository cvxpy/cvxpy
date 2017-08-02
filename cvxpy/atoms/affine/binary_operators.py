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
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import operator as op
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

    def numeric(self, values):
        """Applies the binary operator to the values.
        """
        return reduce(self.OP_FUNC, values)

    def sign_from_args(self):
        """Default to rules for times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])


class MulExpression(BinaryOperator):
    OP_NAME = "*"
    OP_FUNC = op.mul

    def numeric(self, values):
        """Matrix multiplication.
        """
        if self.args[0].is_scalar() or self.args[1].is_scalar():
            return values[0] * values[1]
        else:
            return values[0].dot(values[1])

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.mul_shapes(self.args[0].shape, self.args[1].shape)

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonpos()

    def validate_arguments(self):
        """Validates the dimensions.
        """
        u.shape.mul_shapes(self.args[0].shape, self.args[1].shape)

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Multiply the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.mul_expr(arg_objs[0], arg_objs[1], shape), [])


class RMulExpression(MulExpression):
    """Multiplication by a constant on the right.
    """

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1].is_nonpos()

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Multiply the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.rmul_expr(arg_objs[0], arg_objs[1], shape), [])


class DivExpression(BinaryOperator):
    OP_NAME = "/"
    OP_FUNC = op.__truediv__ if (sys.version_info >= (3, 0)) else op.__div__

    def is_quadratic(self):
        return self.args[0].is_quadratic() and self.args[1].is_constant()

    def is_qpwa(self):
        return self.args[0].is_qpwa() and self.args[1].is_constant()

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1].is_nonpos()

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Multiply the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.div_expr(arg_objs[0], arg_objs[1]), [])
