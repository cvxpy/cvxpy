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
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


def get_diff_mat(dim: int, axis: int) -> sp.csc_matrix:
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

    mat = sp.csc_matrix((val_arr, (row_arr, col_arr)),
                        (dim, dim))
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
    def __init__(self, expr: Expression, axis: int = 0) -> None:
        super(cumsum, self).__init__(expr, axis)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Convolve the two values.
        """
        return np.cumsum(values[0], axis=self.axis)

    def shape_from_args(self) -> Tuple[int, ...]:
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

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
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
        # X = Y[1:,:] - Y[:-1, :]
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
