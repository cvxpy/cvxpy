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
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.abs import abs
from cvxpy.atoms.affine.sum_entries import sum_entries
from numpy import linalg as LA

class norm1(Atom):
    """L1 norm; :math:`\sum_i|x_i|`.

    """
    def __init__(self, x):
        super(norm1, self).__init__(x)

    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the L1 norm of x.
        """
        cols = values[0].shape[1]
        return sum([LA.norm(values[0][:, i], 1) for i in range(cols)])

    def shape_from_args(self):
        """Resolves to a scalar.
        """
        return u.Shape(1, 1)

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arguments and decreasing for negative.
        """
        return [u.monotonicity.SIGNED]

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
        obj, abs_constr = abs.graph_implementation([x], x.size)
        obj, sum_constr = sum_entries.graph_implementation([obj], (1, 1))
        return (obj, abs_constr + sum_constr)
