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

from cvxpy.atoms.atom import Atom
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.constraints.second_order import SOC
import numpy as np
import scipy.sparse as sp
import scipy as scipy


class quad_over_lin(Atom):
    """ :math:`(sum_{ij}X^2_{ij})/y`

    """

    def __init__(self, x, y):
        super(quad_over_lin, self).__init__(x, y)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the sum of the entries of x squared over y.
        """
        return np.square(values[0]).sum()/values[1]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        # y > 0.
        return [self.args[1] >= 0]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = values[0]
        y = values[1]
        if y <= 0:
            return [None, None]
        else:
            # DX = 2X/y, Dy = -||X||^2_2/y^2
            Dy = -np.square(X).sum()/np.square(y)
            Dy = sp.csc_matrix(Dy)
            DX = 2.0*X/y
            DX = np.reshape(DX, (self.args[0].size[0]*self.args[0].size[1], 1))
            DX = scipy.sparse.csc_matrix(DX)
            return [DX, Dy]

    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return (1, 1)

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return (idx == 0) and self.args[idx].is_positive()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return ((idx == 0) and self.args[idx].is_negative()) or (idx == 1)

    def validate_arguments(self):
        """Check dimensions of arguments.
        """
        if not self.args[1].is_scalar():
            raise ValueError("The second argument to quad_over_lin must be a scalar.")

    def is_quadratic(self):
        """Quadratic if x is affine and y is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

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
        x = arg_objs[0]
        y = arg_objs[1]  # Known to be a scalar.
        v = lu.create_var((1, 1))
        two = lu.create_const(2, (1, 1))
        constraints = [SOC(lu.sum_expr([y, v]),
                           [lu.sub_expr(y, v),
                            lu.mul_expr(two, x, x.size)]),
                       lu.create_geq(y)]
        return (v, constraints)
