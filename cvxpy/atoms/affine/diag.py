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
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class diag(AffAtom):
    """Extracts the diagonal from a matrix or makes a vector a matrix diagonal.
    """
    def __init__(self, expr):
        super(diag, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Convolve the two values.
        """
        # Convert values to 1D.
        values = map(intf.from_2D_to_1D, values)
        return np.diag(values[0])

    def validate_arguments(self):
        """Checks that matrix arguments are square.
        """
        rows, cols = self.args[0].size
        if self.args[0].is_matrix() and rows != cols:
            raise ValueError("Argument to diag a vector or square matrix.")

    def shape_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        rows, cols = self.args[0].size
        if self.args[0].is_matrix():
            return u.Shape(rows, 1)
        else:
            return u.Shape()

    def sign_from_args(self):
        """Always unknown.
        """
        return u.Sign.UNKNOWN

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Convolve two vectors.

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
        return (lu.conv(arg_objs[0], arg_objs[1], size), [])
