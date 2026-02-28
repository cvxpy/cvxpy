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
from typing import Tuple

import numpy as np
import scipy.sparse as sp

from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.transpose import swapaxes as expr_swapaxes
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint


class lambda_max(Atom):
    """ Maximum eigenvalue; :math:`\\lambda_{\\max}(A)`.
    """

    def __init__(self, A) -> None:
        super(lambda_max, self).__init__(A)

    def numeric(self, values):
        """Returns the largest eigenvalue of A.

        Requires that A be symmetric.
        """
        return np.linalg.eigvalsh(values[0])[..., -1]

    def _domain(self) -> list[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        A = self.args[0]
        if A.ndim == 2:
            return [A.H == A]
        else:
            if A.is_real():
                return [expr_swapaxes(A, -2, -1) == A]
            else:
                return [expr_swapaxes(conj(A), -2, -1) == A]

    def _grad_matrices(self, A):
        """Compute gradient matrices for all batch elements.

        Returns an array of shape (*batch, n, n) where each (n, n) slice
        is the (sub)gradient of the atom w.r.t. that matrix.
        """
        _, v = np.linalg.eigh(A)
        v_max = v[..., :, -1]  # (..., n)
        return v_max[..., :, np.newaxis] * v_max[..., np.newaxis, :]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        A = values[0]
        D = self._grad_matrices(A)
        total_batch = max(1, int(np.prod(A.shape[:-2])))
        total_size = D.size
        D_flat = D.ravel(order='F')
        all_indices = np.arange(total_size)
        col_indices = all_indices % total_batch
        grad = sp.csc_array(
            (D_flat, (all_indices, col_indices)),
            shape=(total_size, total_batch)
        )
        return [grad]

    def validate_arguments(self) -> None:
        """Verify that the argument A is a square matrix (or batch of square matrices).
        """
        A = self.args[0]
        if A.ndim < 2 or A.shape[-2] != A.shape[-1]:
            raise ValueError("The argument '%s' to lambda_max must resolve to a square matrix."
                             % A.name())

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape[:-2]

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        return (False, False)

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

    @property
    def value(self):
        val = self.args[0].value
        if not np.allclose(val, np.swapaxes(val, -2, -1).conj()):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()
