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
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.error import DCPError
import numpy as np


class multiply(AffAtom):
    """ Multiplies two expressions elementwise.

    The first expression must be constant.
    """

    def __init__(self, lh_const, rh_expr):
        super(multiply, self).__init__(lh_const, rh_expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Multiplies the values elementwise.
        """
        return np.multiply(values[0], values[1])

    def shape_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def sign_from_args(self):
        """Same as times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[0].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[0].is_nonpos()

    def is_quadratic(self):
        """Quadratic if x is quadratic.
        """
        return self.args[1].is_quadratic()

    def is_qpwa(self):
        """Quadratic of PWA if x is QPWA.
        """
        return self.args[1].is_qpwa()

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Multiply the expressions elementwise.

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
        # promote if necessary.
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if lu.is_scalar(arg_objs[0]):
            lhs = lu.promote(arg_objs[0], arg_objs[1].shape)
        elif lu.is_scalar(rhs):
            rhs = lu.promote(rhs, lhs.shape)
        if lu.is_const(lhs):
            return (lu.multiply(lhs, rhs, shape), [])
        elif lu.is_const(rhs):
            return (lu.multiply(rhs, lhs, shape), [])
        else:
            raise DCPError("Product of two non-constant expressions is not "
                           "DCP.")
