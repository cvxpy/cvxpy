"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
import numpy as np


def vstack(arg_list) -> "Vstack":
    """Wrapper on vstack to ensure list argument.
    """
    return Vstack(*arg_list)


class Vstack(AffAtom):
    """ Vertical concatenation """
    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

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
    def validate_arguments(self) -> None:
        model = self.args[0].shape
        # Promote scalars.
        if model == ():
            model = (1,)
        for arg in self.args[1:]:
            arg_shape = arg.shape
            # Promote scalars.
            if arg_shape == ():
                arg_shape = (1,)
            if len(arg_shape) != len(model) or \
               (len(model) > 1 and model[1:] != arg_shape[1:]) or \
               (len(model) <= 1 and model != arg_shape):
                raise ValueError(("All the input dimensions except"
                                  " for axis 0 must match exactly."))

    def graph_implementation(self, arg_objs, shape, data=None):
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
