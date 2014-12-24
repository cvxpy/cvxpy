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
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import numpy as np

def diag(expr):
    """Extracts the diagonal from a matrix or makes a vector a diagonal matrix.

    Parameters
    ----------
    expr : Expression or numeric constant
        A vector or square matrix.

    Returns
    -------
    Expression
        An Expression representing the diagonal vector/matrix.
    """
    expr = AffAtom.cast_to_const(expr)
    if expr.is_vector():
        if expr.size[1] == 1:
            return diag_vec(expr)
        # Convert a row vector to a column vector.
        else:
            expr = reshape(expr, expr.size[1], 1)
            return diag_vec(expr)
    elif expr.size[0] == expr.size[1]:
        return diag_mat(expr)
    else:
        raise ValueError("Argument to diag must be a vector or square matrix.")

class diag_vec(AffAtom):
    """Converts a vector into a diagonal matrix.
    """
    def __init__(self, expr):
        super(diag_vec, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        # Convert values to 1D.
        value = intf.from_2D_to_1D(values[0])
        return np.diag(value)

    def shape_from_args(self):
        """A square matrix.
        """
        rows, _ = self.args[0].size
        return u.Shape(rows, rows)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Convolve two vectors.

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
        return (lu.diag_vec(arg_objs[0]), [])

class diag_mat(AffAtom):
    """Extracts the diagonal from a square matrix.
    """
    def __init__(self, expr):
        super(diag_mat, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Extract the diagonal from a square matrix constant.
        """
        # The return type in numpy versions < 1.10 was ndarray.
        v = np.diag(values[0])
        if isinstance(v, np.matrix):
            v = v.A[0]
        return v

    def shape_from_args(self):
        """A column vector.
        """
        rows, _ = self.args[0].size
        return u.Shape(rows, 1)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Extracts the diagonal of a matrix.

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
        return (lu.diag_mat(arg_objs[0]), [])
