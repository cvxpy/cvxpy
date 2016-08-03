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
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class max_entries(AxisAtom):
    """:math:`\max_{i,j}\{X_{i,j}\}`.
    """

    def __init__(self, x, axis=None):
        super(max_entries, self).__init__(x, axis=axis)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the largest entry in x.
        """
        return values[0].max(axis=self.axis)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray or None.
        """
        # Grad: 1 for a largest index.
        value = np.matrix(value).A.ravel(order='F')
        idx = np.argmax(value)
        D = np.zeros((value.size, 1))
        D[idx] = 1
        return D

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Same as argument.
        return (self.args[0].is_positive(), self.args[0].is_negative())

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
        return True

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return False

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
        axis = data[0]
        if axis is None:
            t = lu.create_var((1, 1))
            promoted_t = lu.promote(t, arg_objs[0].size)
        elif axis == 0:
            t = lu.create_var((1, arg_objs[0].size[1]))
            const_size = (arg_objs[0].size[0], 1)
            ones = lu.create_const(np.ones(const_size), const_size)
            promoted_t = lu.mul_expr(ones, t, arg_objs[0].size)
        else:  # axis == 1
            t = lu.create_var((arg_objs[0].size[0], 1))
            const_size = (1, arg_objs[0].size[1])
            ones = lu.create_const(np.ones(const_size), const_size)
            promoted_t = lu.rmul_expr(t, ones, arg_objs[0].size)

        constraints = [lu.create_leq(arg_objs[0], promoted_t)]
        return (t, constraints)
