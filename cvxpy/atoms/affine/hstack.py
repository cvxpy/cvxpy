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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
import numpy as np


class hstack(AffAtom):
    """ Horizontal concatenation """
    # Returns the hstack of the values.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.hstack(values)

    # The shape is the common height and the sum of the widths.
    def size_from_args(self):
        cols = sum(arg.size[1] for arg in self.args)
        rows = self.args[0].size[0]
        return (rows, cols)

    # All arguments must have the same height.
    def validate_arguments(self):
        arg_cols = [arg.size[0] for arg in self.args]
        if max(arg_cols) != min(arg_cols):
            raise TypeError(("All arguments to hstack must have "
                             "the same number of rows."))

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Stack the expressions horizontally.

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
        return (lu.hstack(arg_objs, size), [])
