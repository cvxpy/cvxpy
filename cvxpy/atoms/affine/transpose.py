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


class transpose(AffAtom):
    """Transpose an expression.
    """

    def __init__(self, expr, axes=None) -> None:
        self.axes = axes
        super(AffAtom, self).__init__(expr)

    # The string representation of the atom.
    def name(self) -> str:
        return "%s.T" % self.args[0]

    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.transpose(values[0], axes=self.axes)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.args[0].is_symmetric()

    def is_hermitian(self) -> bool:
        """Is the expression Hermitian?
        """
        return self.args[0].is_hermitian()

    def shape_from_args(self):
        """Returns the shape of the transpose expression.
        """
        return self.args[0].shape[::-1]

    def get_data(self):
        """ Returns the axes for transposition.
        """
        return [self.axes]

    def graph_implementation(self, arg_objs, shape, data=None):
        """Create a new variable equal to the argument transposed.

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
        # TODO(akshakya): This will need to be updated when we add support
        # for >2D arrays.
        return (lu.transpose(arg_objs[0]), [])
