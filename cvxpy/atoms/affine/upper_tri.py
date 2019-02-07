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

from cvxpy.atoms.affine.affine_atom import AffAtom
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
        value = np.zeros(self.shape[0])
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
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError(
                "Argument to upper_tri must be a square matrix."
            )

    def shape_from_args(self):
        """A vector.
        """
        rows, cols = self.args[0].shape
        return (rows*(cols-1)//2, 1)

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return True

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Vectorized strictly upper triagonal entries.

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
        return (lu.upper_tri(arg_objs[0]), [])
