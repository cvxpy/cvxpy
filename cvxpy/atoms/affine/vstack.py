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


def vstack(arg_list):
    """Wrapper on vstack to ensure list argument.
    """
    return Vstack(*arg_list)


class Vstack(AffAtom):
    """ Vertical concatenation """
    # Returns the vstack of the values.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.vstack(values)

    # The shape is the common width and the sum of the heights.
    def shape_from_args(self):
        self.args[0].shape
        if self.args[0].ndim == 0:
            return (len(self.args), 1)
        elif self.args[0].ndim == 1:
            return (len(self.args), self.args[0].shape[0])
        else:
            rows = sum(arg.shape[0] for arg in self.args)
            return (rows,) + self.args[0].shape[1:]

    # All arguments must have the same width.
    def validate_arguments(self):
        model = self.args[0].shape
        for arg in self.args[1:]:
            if len(arg.shape) != len(model) or \
               (len(model) > 1 and model[1:] != arg.shape[1:]) or \
               (len(model) <= 1 and model != arg.shape):
                raise ValueError(("All the input dimensions except"
                                  " for axis 0 must match exactly."))

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Stack the expressions vertically.

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
        return (lu.vstack(arg_objs, shape), [])
