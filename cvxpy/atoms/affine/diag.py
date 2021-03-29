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
from cvxpy.atoms.affine.vec import vec
from typing import Union

import cvxpy.lin_ops.lin_utils as lu
import numpy as np


def diag(expr) -> Union["diag_mat", "diag_vec"]:
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
        return diag_vec(vec(expr))
    elif expr.ndim == 2 and expr.shape[0] == expr.shape[1]:
        return diag_mat(expr)
    else:
        raise ValueError("Argument to diag must be a vector or square matrix.")


class diag_vec(AffAtom):
    """Converts a vector into a diagonal matrix.
    """

    def __init__(self, expr) -> None:
        super(diag_vec, self).__init__(expr)

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
        return np.diag(values[0])

    def shape_from_args(self):
        """A square matrix.
        """
        rows = self.args[0].shape[0]
        return (rows, rows)

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        return True

    def is_hermitian(self) -> bool:
        """Is the expression symmetric?
        """
        return True

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        return self.is_nonneg()

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        return self.is_nonpos()

    def graph_implementation(self, arg_objs, shape, data=None):
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
        return (lu.diag_vec(arg_objs[0]), [])


class diag_mat(AffAtom):
    """Extracts the diagonal from a square matrix.
    """

    def __init__(self, expr) -> None:
        super(diag_mat, self).__init__(expr)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    @AffAtom.numpy_numeric
    def numeric(self, values):
        """Extract the diagonal from a square matrix constant.
        """
        # The return type in numpy versions < 1.10 was ndarray.
        return np.diag(values[0])

    def shape_from_args(self):
        """A column vector.
        """
        rows, _ = self.args[0].shape
        return (rows,)

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        return self.args[0].is_nonneg() or self.args[0].is_psd()

    def graph_implementation(self, arg_objs, shape, data=None):
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
        return (lu.diag_mat(arg_objs[0]), [])
