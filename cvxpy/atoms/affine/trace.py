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
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

class trace(AffAtom):
    """The sum of the diagonal entries of a matrix.

    Attributes
    ----------
    expr : CVXPY Expression
        The expression to sum the diagonal of.
    """

    def __init__(self, expr):
        super(trace, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Sums the diagonal entries.
        """
        return np.trace(values[0])

    def validate_arguments(self):
        """Checks that the argument is a square matrix.
        """
        rows, cols = self.args[0].size
        if not rows == cols:
            raise ValueError("Argument to trace must be a square matrix.")

    def shape_from_args(self):
        """Always scalar.
        """
        return u.Shape(1, 1)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Sum the diagonal entries of the linear expression.

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
        return (lu.trace(arg_objs[0]), [])
