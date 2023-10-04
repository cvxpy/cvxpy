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
from scipy import linalg as LA
from scipy.special import entr

from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint


class von_neumann_entr(Atom):
    """
    Computes the Von Neumann Entropy of the positive-definite matrix
    :math:`X\\in\\mathbb{S}^n_{+}`

        .. math::
            -\\operatorname{tr}(X \\operatorname{logm}(X))

    where :math:`\\operatorname{tr}` is the trace and
    :math:`\\operatorname{logm}` is the matrix logarithm

    | May alternatively be expressed as:

        .. math::
            \\texttt{von_neumann_entr}(X) = -\\textstyle\\sum_{i=1}^n \\lambda_i \\log \\lambda_i

    | where :math:`\\lambda_{i}` are the eigenvalues of :math:`X`
    This atom does not enforce :math:`\\operatorname{tr}(X) = 1`
    as is expected in applications from quantum mechanics.

    Parameters
    ----------
    X : Expression or numeric
        A PSD matrix
    """

    def __init__(self, X, quad_approx: Tuple[int, int] = ()) -> None:
        # TODO: add a check that N is symmetric/Hermitian.
        self.quad_approx = quad_approx
        super(von_neumann_entr, self).__init__(X)

    def numeric(self, values):
        N = values[0]
        if hasattr(N, 'value'):
            N = N.value  # assume this is an ndarray
        w = LA.eigvalsh(N)
        val = np.sum(entr(w))
        return val

    def validate_arguments(self) -> None:
        """Verify that the argument A is PSD.
        """
        N = self.args[0]
        if N.size > 1:
            if N.ndim != 2 or N.shape[0] != N.shape[1]:
                raise ValueError('Argument must be a square matrix.')

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        return False

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the shape of the expression.
        """
        return tuple()

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        return True

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return False

    def get_data(self):
        return [self.quad_approx]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        # N = values[0]
        # L = cholesky(N)
        # derivative = 2*(L + L * logm(np.dot(L.T, L)))
        # TODO: have to wrap derivative around scipy CSC sparse matrices
        #  compare to log_det atom.
        raise ValueError()

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        return [self.args[0] >> 0]
