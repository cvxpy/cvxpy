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

class reshape(AffAtom):
    """ Reshapes the expression.

    Vectorizes the expression then unvectorizes it into the new shape.
    The entries are stored in column-major order.
    """
    def __init__(self, expr, rows, cols):
        self.rows = rows
        self.cols = cols
        super(reshape, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Reshape the value.
        """
        return np.reshape(values[0], (self.rows, self.cols), "F")

    def validate_arguments(self):
        """Checks that the new shape has the same number of entries as the old.
        """
        old_len = self.args[0].size[0]*self.args[0].size[1]
        new_len = self.rows*self.cols
        if not old_len == new_len:
            raise ValueError(
                "Invalid reshape dimensions (%i, %i)." % (self.rows, self.cols)
            )

    def shape_from_args(self):
        """Returns the shape from the rows, cols arguments.
        """
        return u.Shape(self.rows, self.cols)

    def get_data(self):
        """Returns info needed to reconstruct the expression besides the args.
        """
        return [self.rows, self.cols]

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
        return (lu.reshape(arg_objs[0], size), [])
