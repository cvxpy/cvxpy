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
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression


class upper_tri(AffAtom):
    """
    The vectorized strictly upper-triangular entries.

    The vectorization is performed by concatenating (partial) rows.
    For example, if

    ::

        A = np.array([[10, 11, 12, 13],
                      [14, 15, 16, 17],
                      [18, 19, 20, 21],
                      [22, 23, 24, 25]])

    then we have

    ::

        upper_tri(A).value == np.array([11, 12, 13, 16, 17, 21])

    """

    def __init__(self, expr) -> None:
        super(upper_tri, self).__init__(expr)

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """
        Vectorize the strictly upper triangular entries.
        """
        upper_idx = np.triu_indices(n=values[0].shape[0], k=1, m=values[0].shape[1])
        return values[0][upper_idx]

    def validate_arguments(self) -> None:
        """Checks that the argument is a square matrix.
        """
        if not self.args[0].ndim == 2 or self.args[0].shape[0] != self.args[0].shape[1]:
            raise ValueError(
                "Argument to upper_tri must be a square matrix."
            )

    def shape_from_args(self) -> Tuple[int, int]:
        """A vector.
        """
        rows, cols = self.args[0].shape
        return (rows*(cols-1)//2, 1)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Vectorized strictly upper triangular entries.

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


def vec_to_upper_tri(expr, strict: bool = False):
    """Reshapes a vector into an upper triangular matrix in
    row-major order. The strict argument specifies whether an upper or a strict upper triangular
    matrix should be returned.
    Inverts cp.upper_tri.
    """
    expr = Expression.cast_to_const(expr)

    if not expr.is_vector():
        raise ValueError("The input must be a vector.")
    if expr.ndim != 1:
        expr = vec(expr, order='F')

    ell = expr.shape[0]
    if strict:
        # n * (n-1)/2 == ell
        n = ((8 * ell + 1) ** 0.5 + 1) // 2
    else:
        # n * (n+1)/2 == ell
        n = ((8 * ell + 1) ** 0.5 - 1) // 2
    n = int(n)
    if not (n * (n + 1) // 2 == ell or n * (n - 1) // 2 == ell):
        raise ValueError("The size of the vector must be a triangular number.")

    """
    Initialize a coefficient matrix P that creates an upper triangular matrix when 
    multiplied with a variable vector expr.
    That is, (P @ expr).reshape((n, n)) is an upper triangular matrix.
    """
    k = 1 if strict else 0
    row_idx, col_idx = np.triu_indices(n, k=k)
    P_rows = n * row_idx + col_idx
    P_cols = np.arange(ell)
    P_vals = np.ones(P_cols.size)
    P = sp.csc_matrix((P_vals, (P_rows, P_cols)), shape=(n * n, ell))
    return reshape(P @ expr, (n, n), order='F').T


def upper_tri_to_full(n: int) -> sp.csc_matrix:
    """
    Returns a coefficient matrix A that creates a symmetric matrix when
    multiplied with a variable vector v.
    That is, (A @ v).reshape((n, n)) is a symmetric matrix.

    Parameters
    ----------
    n : int
        The length of the matrix.

    Returns
    -------
    sp.csc_matrix
        The coefficient matrix.
    """
    entries = n*(n+1)//2

    # Initialize row and col indices from upper triangular matrix
    rows, cols = np.triu_indices(n)

    # Mask for the symmetric part when i != j
    mask = rows != cols

    row_idx = np.concatenate([rows * n + cols, cols[mask] * n + rows[mask]])
    col_idx = np.concatenate([np.arange(entries), np.arange(entries)[mask]])
    values = np.ones(col_idx.size, dtype=float)

    # Construct and return the sparse matrix
    return sp.csc_matrix((values, (row_idx, col_idx)), shape=(n * n, entries))
