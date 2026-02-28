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

import numpy as np

from cvxpy.atoms.lambda_max import lambda_max


class lambda_sum_largest(lambda_max):
    """Sum of the largest k eigenvalues.
    """
    _allow_complex = True

    def __init__(self, X, k) -> None:
        self.k = k
        super(lambda_sum_largest, self).__init__(X)

    def validate_arguments(self) -> None:
        """Verify that the argument A is square.
        """
        X = self.args[0]
        if X.ndim < 2 or X.shape[-2] != X.shape[-1]:
            raise ValueError("First argument must be a square matrix.")
        elif self.k <= 0:
            raise ValueError("Second argument must be a positive number.")

    def numeric(self, values):
        """Returns the sum of the k largest eigenvalues of A.

        Requires that A be symmetric.
        """
        # eigvalsh returns eigenvalues sorted ascending along the last axis.
        eigs = np.linalg.eigvalsh(values[0])
        k_floor = int(np.floor(self.k))
        k_frac = self.k - k_floor
        result = np.zeros(eigs.shape[:-1]) if eigs.ndim > 1 else 0.0
        if k_floor > 0:
            result = result + eigs[..., -k_floor:].sum(axis=-1)
        if k_frac > 0 and k_floor < eigs.shape[-1]:
            result = result + k_frac * eigs[..., -(k_floor + 1)]
        return result

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]

    # _grad is inherited from lambda_max; only _grad_matrices is overridden.

    def _grad_matrices(self, A):
        """Compute gradient matrices for all batch elements.

        Returns an array of shape (*batch, n, n).
        """
        k = self.k
        k_floor = int(np.floor(k))
        k_frac = k - k_floor
        n = A.shape[-1]
        _, v = np.linalg.eigh(A)
        # Eigenvalues sorted ascending; largest k are the last k columns.
        D = np.zeros(A.shape)
        if k_floor > 0:
            V_top = v[..., :, -k_floor:]  # (..., n, k_floor)
            D = D + V_top @ np.swapaxes(V_top, -2, -1)
        if k_frac > 0 and k_floor < n:
            v_next = v[..., :, -(k_floor + 1)]  # (..., n)
            D = D + k_frac * (v_next[..., :, np.newaxis] * v_next[..., np.newaxis, :])
        return D

    @property
    def value(self):
        val = self.args[0].value
        if not np.allclose(val, np.swapaxes(val, -2, -1).conj()):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()
