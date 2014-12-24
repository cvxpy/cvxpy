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

class upper_tri(AffAtom):
    """The vectorized strictly upper triagonal entries.
    """
    def __init__(self, expr):
        super(upper_tri, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Vectorize the upper triagonal entries.
        """
        value = np.zeros(self.size[0])
        count = 0
        for i in range(values[0].shape[0]):
            for j in range(values[0].shape[1]):
                if i < j:
                    value[count] = values[0][i, j]
                    count += 1
        return value

    def validate_arguments(self):
        """Checks that the argument is a square matrix.
        """
        if not self.args[0].size[0] == self.args[0].size[1]:
            raise ValueError(
                "Argument to upper_tri must be a square matrix."
            )

    def shape_from_args(self):
        """A vector.
        """
        rows, cols = self.args[0].size
        return u.Shape(rows*(cols-1)//2, 1)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Vectorized strictly upper triagonal entries.

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
        return (lu.upper_tri(arg_objs[0]), [])
