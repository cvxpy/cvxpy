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

from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.expressions.variable import Variable
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
import scipy.sparse as sp


def get_diff_mat(dim, axis):
    """Return a sparse matrix representation of first order difference operator.

    Parameters
    ----------
    dim : int
       The length of the matrix dimensions.
    axis : int
       The axis to take the difference along.

    Returns
    -------
    SciPy CSC matrix
        A square matrix representing first order difference.
    """
    # Construct a sparse matrix representation.
    val_arr = []
    row_arr = []
    col_arr = []
    for i in range(dim):
        val_arr.append(1.)
        row_arr.append(i)
        col_arr.append(i)
        if i > 0:
            val_arr.append(-1.)
            row_arr.append(i)
            col_arr.append(i-1)

    mat = sp.coo_matrix((val_arr, (row_arr, col_arr)),
                        (dim, dim)).tocsc()
    if axis == 0:
        return mat
    else:
        return mat.T


class cumsum(AffAtom, AxisAtom):
    """Cumulative sum.

    Attributes
    ----------
    expr : CVXPY expression
        The expression being summed.
    axis : int
        The axis to sum across if 2D.
    """
    def __init__(self, expr, axis=0):
        super(cumsum, self).__init__(expr, axis)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Convolve the two values.
        """
        return np.cumsum(values[0], axis=self.axis)

    def shape_from_args(self):
        """The same as the input.
        """
        return self.args[0].shape

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # TODO inefficient
        dim = values[0].shape[self.axis]
        mat = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(i+1):
                mat[i, j] = 1
        var = Variable(self.args[0].shape)
        if self.axis == 0:
            grad = MulExpression(mat, var)._grad(values)[1]
        else:
            grad = MulExpression(var, mat.T)._grad(values)[0]
        return [grad]

    def get_data(self):
        """Returns the axis being summed.
        """
        return [self.axis]

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        """Cumulative sum via difference matrix.

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
        # Implicit O(n) definition:
        # X = Y[:1,:] - Y[1:,:]
        Y = lu.create_var(shape)
        axis = data[0]
        dim = shape[axis]
        diff_mat = get_diff_mat(dim, axis)
        diff_mat = lu.create_const(diff_mat, (dim, dim), sparse=True)
        if axis == 0:
            diff = lu.mul_expr(diff_mat, Y)
        else:
            diff = lu.rmul_expr(Y, diff_mat)
        return (Y, [lu.create_eq(arg_objs[0], diff)])
