"""
Copyright 2017 Steven Diamond

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


class hstack(AffAtom):
    """ Horizontal concatenation """
    # Can take a single list as input.
    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        super(hstack, self).__init__(*args)

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
