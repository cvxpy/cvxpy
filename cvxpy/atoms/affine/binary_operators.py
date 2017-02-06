"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
        """Multiplication checks for 1D array * 1D array.
        """
        return reduce(self.OP_FUNC, values)

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return u.shape.mul_shapes(self.args[0].size, self.args[1].size)

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_negative()

    def validate_arguments(self):
        """Validates the dimensions.
        """
        u.shape.mul_shapes(self.args[0].size, self.args[1].size)

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

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1].is_negative()

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
    OP_FUNC = op.__truediv__ if (sys.version_info >= (3, 0)) else op.__div__

    def is_quadratic(self):
        return self.args[0].is_quadratic() and self.args[1].is_constant()

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return self.args[0].size

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1].is_negative()

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
