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
from cvxpy.atoms.affine.promote import promote
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.error import DCPError
import numpy as np


class multiply(AffAtom):
    """ Multiplies two expressions elementwise.
    """

    def __init__(self, lh_expr, rh_expr):
        lh_expr = multiply.cast_to_const(lh_expr)
        rh_expr = multiply.cast_to_const(rh_expr)
        if lh_expr.is_scalar() and not rh_expr.is_scalar():
            lh_expr = promote(lh_expr, rh_expr.shape)
        elif rh_expr.is_scalar() and not lh_expr.is_scalar():
            rh_expr = promote(rh_expr, lh_expr.shape)
        super(multiply, self).__init__(lh_expr, rh_expr)

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

    def is_atom_convex(self):
        """Multiplication is convex (affine) in its arguments only if one of
           the arguments is constant.
        """
        return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self):
        """If the multiplication atom is convex, then it is affine.
        """
        return self.is_atom_convex()

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1-idx].is_nonneg()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1-idx].is_nonpos()

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
            (LinOp for objective, list of exprraints)
        """
        # promote if necessary.
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if lu.is_const(lhs):
            return (lu.multiply(lhs, rhs), [])
        elif lu.is_const(rhs):
            return (lu.multiply(rhs, lhs), [])
        else:
            raise DCPError("Product of two non-exprant expressions is not "
                           "DCP.")
