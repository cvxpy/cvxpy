"""
Copyright 2022, the CVXPY authors

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
from numpy import linalg as LA

import cvxpy.settings as s
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint


class tr_inv(Atom):
    r"""
    :math:`\mathrm{tr}\left(X^{-1} \right),`
    where :math:`X` is positive definite.
    """

    def __init__(self, X) -> None:
        super(tr_inv, self).__init__(X)

    def numeric(self, values):
        """Returns the trinv of positive definite matrix X.

        For positive definite matrix X, this is the trace of inverse of X.
        """
        # if values[0] isn't Hermitian then return np.inf
        if (LA.norm(values[0] - values[0].T.conj()) >= 1e-8):
            return np.inf
        # take symmetric part of the input to enhance numerical stability
        symm = (values[0] + values[0].T)/2
        eigVal = LA.eigvalsh(symm)
        if min(eigVal) <= 0:
            return np.inf
        return np.sum(eigVal**-1)

    # The shape of argument must be square.
    def validate_arguments(self) -> None:
        X = self.args[0]
        if len(X.shape) == 1 or X.shape[0] != X.shape[1]:
            raise TypeError("The argument to tr_inv must be a square matrix.")

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

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        X = values[0]
        eigen_val = LA.eigvalsh(X)
        if np.min(eigen_val) > 0:
            # Grad: -X^{-2}.T
            D = np.linalg.inv(X).T
            D = - D @ D
            return [sp.csc_matrix(D.ravel(order='F')).T]
        # Outside domain.
        else:
            return [None]

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >> 0]

    @property
    def value(self) -> float:
        if not np.allclose(self.args[0].value,
                           self.args[0].value.T.conj(),
                           rtol=s.ATOM_EVAL_TOL,
                           atol=s.ATOM_EVAL_TOL):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()
