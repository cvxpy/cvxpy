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
from cvxpy.expressions.expression import Expression
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.reshape import reshape
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
from scipy.sparse import csc_matrix


class upper_tri(AffAtom):
    """The vectorized strictly upper-triagonal entries.

    The vectorization is performed by concatenating (partial) rows.
    For example, if
    ```
    A = np.array([[10, 11, 12, 13],
                  [14, 15, 16, 17],
                  [18, 19, 20, 21],
                  [22, 23, 24, 25]])
    ```
    then we have
    ```
    upper_tri(A).value == np.array([11, 12, 13, 16, 17, 21])
    ```
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

    def graph_implementation(self, arg_objs, shape, data=None):
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


def vec_to_upper_tri(expr, strict=False):
    expr = Expression.cast_to_const(expr)
    ell = expr.shape[0]
    if strict:
        # n * (n-1)/2 == ell
        n = ((8 * ell + 1) ** 0.5 + 1) // 2
    else:
        # n * (n+1)/2 == ell
        n = ((8 * ell + 1) ** 0.5 - 1) // 2
    n = int(n)
    # form a matrix P, of shape (n**2, ell).
    #       the i-th block of n rows of P gives the entries of the i-th row
    #       of the upper-triangular matrix associated with expr.
    # compute expr2 = P @ expr
    # compute expr3 = reshape(expr2, shape=(n, n)).T
    #   expr3 is the matrix formed by reading length-n blocks of expr2,
    #   and letting each block form a row of expr3.
    P_rows = []
    P_row = 0
    for mat_row in range(n):
        entries_in_row = n - mat_row
        if strict:
            entries_in_row -= 1
        P_row += n - entries_in_row  # these are zeros
        P_rows.extend(range(P_row, P_row + entries_in_row))
        P_row += entries_in_row
    P_cols = np.arange(ell)
    P_vals = np.ones(P_cols.size)
    P = csc_matrix((P_vals, (P_rows, P_cols)), shape=(n**2, ell))
    expr2 = P @ expr
    expr3 = reshape(expr2, (n, n)).T
    return expr3
