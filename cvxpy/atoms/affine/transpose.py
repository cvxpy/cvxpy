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
import cvxpy.lin_ops.lin_utils as lu
import numpy as np


class transpose(AffAtom):
    """Transpose an expression.
    """

    def __init__(self, expr, axes=None):
        self.axes = axes
        super(AffAtom, self).__init__(expr)

    # The string representation of the atom.
    def name(self):
        return "%s.T" % self.args[0]

    # Returns the transpose of the given value.
    @AffAtom.numpy_numeric
    def numeric(self, values):
        return np.transpose(values[0], axes=self.axes)

    def is_symmetric(self):
        """Is the expression symmetric?
        """
        return self.args[0].is_symmetric()

    def is_hermitian(self):
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

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
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
        # for >2D ararys.
        return (lu.transpose(arg_objs[0]), [])
