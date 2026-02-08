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
from scipy import linalg as LA

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
        A = values[0]
        if A.ndim == 2:
            lo = hi = A.shape[0] - 1
            return LA.eigvalsh(A, subset_by_index=(lo, hi))[0]
        else:
            batch_shape = A.shape[:-2]
            result = np.empty(batch_shape)
            for idx in np.ndindex(batch_shape):
                mat = A[idx]
                n = mat.shape[0]
                result[idx] = LA.eigvalsh(mat, subset_by_index=(n - 1, n - 1))[0]
            return result

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        A = self.args[0]
        if A.ndim == 2:
            return [A.H == A]
        else:
            if A.is_real():
                return [expr_swapaxes(A, -2, -1) == A]
            else:
                from cvxpy.atoms.affine.conj import conj
                return [expr_swapaxes(conj(A), -2, -1) == A]

    def _single_matrix_grad(self, mat):
        """Compute the gradient matrix for a single 2D symmetric matrix."""
        _, v = LA.eigh(mat)
        v_max = v[:, -1]
        return np.outer(v_max, v_max)

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        A = values[0]
        if A.ndim == 2:
            D = self._single_matrix_grad(A)
            return [sp.csc_array([D.ravel(order='F')]).T]
        else:
            batch_shape = A.shape[:-2]
            n = A.shape[-1]
            total_batch = int(np.prod(batch_shape))
            n2 = n * n
            rows = []
            cols = []
            vals = []
            # F-order flat index: batch_fi + total_batch * matrix_fi
            for idx in np.ndindex(batch_shape):
                D = self._single_matrix_grad(A[idx])
                D_flat = D.ravel(order='F')
                nz = np.nonzero(D_flat)[0]
                batch_fi = np.ravel_multi_index(
                    idx, batch_shape, order='F'
                )
                rows.extend(batch_fi + total_batch * nz)
                cols.extend([batch_fi] * len(nz))
                vals.extend(D_flat[nz])
            grad = sp.csc_array(
                (vals, (rows, cols)),
                shape=(total_batch * n2, total_batch)
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
