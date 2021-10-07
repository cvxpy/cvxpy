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

from functools import wraps
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp
from numpy import linalg as LA

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.constraints.constraint import Constraint


class MatrixFrac(Atom):
    """ tr X.T*P^-1*X """
    _allow_complex = True

    def __init__(self, X, P) -> None:
        super(MatrixFrac, self).__init__(X, P)

    def numeric(self, values):
        """Returns tr X.T*P^-1*X.
        """
        # TODO raise error if not invertible?
        X = values[0]
        P = values[1]
        if self.args[0].is_complex():
            product = np.conj(X).T.dot(LA.inv(P)).dot(X)
        else:
            product = X.T.dot(LA.inv(P)).dot(X)
        return product.trace() if len(product.shape) == 2 else product

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[1] >> 0]

    def _grad(self, values):
        """
        Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = np.array(values[0])
        if X.ndim == 1:
            X = X[:, None]
        P = np.array(values[1])
        try:
            P_inv = LA.inv(P)
        except LA.LinAlgError:
            return [None, None]
        # partial_X = (P^-1+P^-T)X
        # partial_P = - (P^-1 * X * X^T * P^-1)^T
        else:
            DX = np.dot(P_inv+np.transpose(P_inv), X)
            DX = DX.T.ravel(order='F')
            DX = sp.csc_matrix(DX).T

            DP = np.dot(P_inv, X)
            DP = np.dot(DP, X.T)
            DP = np.dot(DP, P_inv)
            DP = -DP.T
            DP = sp.csc_matrix(DP.T.ravel(order='F')).T
            return [DX, DP]

    def validate_arguments(self) -> None:
        """Checks that the dimensions of x and P match.
        """
        X = self.args[0]
        P = self.args[1]
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError(
                "The second argument to matrix_frac must be a square matrix."
            )
        elif X.shape[0] != P.shape[0]:
            raise ValueError(
                "The arguments to matrix_frac have incompatible dimensions."
            )

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return tuple()

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return True

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def is_quadratic(self) -> bool:
        """Quadratic if x is affine and P is constant.
        """
        return self.args[0].is_affine() and self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        """Quadratic of piecewise affine if x is PWL and P is constant.
        """
        return self.args[0].is_pwl() and self.args[1].is_constant()


@wraps(MatrixFrac)
def matrix_frac(X, P):
    if isinstance(P, np.ndarray):
        invP = LA.inv(P)
        return QuadForm(X, (invP + np.conj(invP).T) / 2.0)
    else:
        return MatrixFrac(X, P)
