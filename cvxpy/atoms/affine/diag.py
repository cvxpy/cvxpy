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
from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.constraint import Constraint


def diag(expr, k: int = 0) -> Union["diag_mat", "diag_vec"]:
    """Extracts the diagonal from a matrix or makes a vector a diagonal matrix.

    Parameters
    ----------
    expr : Expression or numeric constant
        A vector or square matrix.

    k : int
        Diagonal in question. The default is 0.
        Use k>0 for diagonals above the main diagonal,
        and k<0 for diagonals below the main diagonal.

    Returns
    -------
    Expression
        An Expression representing the diagonal vector/matrix.
    """
    expr = AffAtom.cast_to_const(expr)
    if expr.is_vector():
        return diag_vec(vec(expr, order='F'), k)
    elif expr.ndim == 2 and expr.shape[0] == expr.shape[1]:
        assert abs(k) < expr.shape[0], "Offset out of bounds."
        return diag_mat(expr, k)
    else:
        raise ValueError("Argument to diag must be a 1-d array or 2-d square array.")


class diag_vec(AffAtom):
    """Converts a vector into a diagonal matrix."""

    def __init__(self, expr, k: int = 0) -> None:
        self.k = k
        super(diag_vec, self).__init__(expr)

    def get_data(self) -> list[int]:
        return [self.k]
    
    def validate_arguments(self) -> None:
        """Checks that the argument is a vector.
        """
        if not self.args[0].ndim == 1:
            raise ValueError(
                "Argument to diag_vec must be a 1-d array"
            )
        
    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def numeric(self, values):
        """Convert the vector constant into a diagonal matrix.
        """
        val = values[0]
        # Handle batched values: create diagonal matrices for each batch element
        batch_ndim = np.ndim(val) - len(self.args[0].shape)
        if batch_ndim > 0:
            batch_shape = val.shape[:-1]
            vec_len = val.shape[-1]
            n = vec_len + abs(self.k)
            result = np.zeros(batch_shape + (n, n), dtype=val.dtype)
            # Set diagonal entries using advanced indexing
            diag_indices = np.arange(vec_len)
            if self.k >= 0:
                result[..., diag_indices, diag_indices + self.k] = val
            else:
                result[..., diag_indices - self.k, diag_indices] = val
            return result
        return np.diag(val, k=self.k)

    def shape_from_args(self) -> Tuple[int, int]:
        """A square matrix.
        """
        rows = self.args[0].shape[0] + abs(self.k)
        return (rows, rows)

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return self.k == 0

    def is_hermitian(self) -> bool:
        """Is the expression hermitian?
        """
        return self.k == 0

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        return self.is_nonneg() and self.k == 0

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        return self.is_nonpos() and self.k == 0

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Convolve two vectors.

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
        return (lu.diag_vec(arg_objs[0], self.k), [])


class diag_mat(AffAtom):
    """Extracts the diagonal from a square matrix.
    """

    def __init__(self, expr, k: int = 0) -> None:
        self.k = k
        super(diag_mat, self).__init__(expr)
    
    def get_data(self) -> list[int]:
        return [self.k]

    def validate_arguments(self) -> None:
        """Checks that the argument is a square matrix.
        """
        if not self.args[0].ndim == 2:
            raise ValueError(
                "Argument to diag_mat must be a 2-d array."
            )

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    @AffAtom.numpy_numeric
    def numeric(self, values) -> np.ndarray:
        """Extract the diagonal from a square matrix constant."""
        val = values[0]
        # Handle batched values: extract diagonal along last two axes
        batch_ndim = np.ndim(val) - len(self.args[0].shape)
        if batch_ndim > 0:
            return np.diagonal(val, offset=self.k, axis1=-2, axis2=-1)
        return np.diag(val, k=self.k)

    def shape_from_args(self) -> Tuple[int]:
        """A column vector."""
        rows, _ = self.args[0].shape
        rows -= abs(self.k)
        return (rows,)

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?"""
        return (self.args[0].is_nonneg() or self.args[0].is_psd()) and self.k == 0

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Extracts the diagonal of a matrix.

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
        return (lu.diag_mat(arg_objs[0], self.k), [])
