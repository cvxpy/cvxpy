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


def hstack(arg_list) -> "Hstack":
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
    def is_atom_log_log_convex(self) -> bool:
        return True

    def is_atom_log_log_concave(self) -> bool:
        return True

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
    def validate_arguments(self) -> None:
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

    def graph_implementation(self, arg_objs, shape, data=None):
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
