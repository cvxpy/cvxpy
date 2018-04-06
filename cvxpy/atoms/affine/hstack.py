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


def hstack(arg_list):
    """Horizontal concatenation of an arbitrary number of Expressions.

    Parameters
    ----------
    arg_list : list of Expression
        The Expressions to concatenate.
    """
    arg_list = [AffAtom.cast_to_const(arg) for arg in arg_list]
    for idx, arg in enumerate(arg_list):
        if arg.ndim == 0:
            arg_list[idx] = arg.flatten()
    return Hstack(*arg_list)


class Hstack(AffAtom):
    """ Horizontal concatenation """
    # Returns the hstack of the values.
    def numeric(self, values):
        return np.hstack(values)

    # The shape is the common width and the sum of the heights.
    def shape_from_args(self):
        if self.args[0].ndim == 1:
            return (sum(arg.size for arg in self.args),)
        else:
            cols = sum(arg.shape[1] for arg in self.args)
            return (self.args[0].shape[0], cols) + self.args[0].shape[2:]

    # All arguments must have the same width.
    def validate_arguments(self):
        model = self.args[0].shape
        error = ValueError(("All the input dimensions except"
                            " for axis 1 must match exactly."))
        for arg in self.args[1:]:
            if len(arg.shape) != len(model):
                raise error
            elif len(model) > 1:
                for i in range(len(model)):
                    if i != 1 and arg.shape[i] != model[i]:
                        raise error

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Stack the expressions horizontally.

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
        return (lu.hstack(arg_objs, shape), [])
