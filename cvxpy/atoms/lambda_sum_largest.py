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
import scipy.sparse as sp
from scipy import linalg as LA

from cvxpy.atoms.lambda_max import lambda_max
from cvxpy.atoms.sum_largest import sum_largest


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
        A = values[0]
        if A.ndim == 2:
            eigs = LA.eigvalsh(A)
            return sum_largest(eigs, self.k).value
        else:
            batch_shape = A.shape[:-2]
            result = np.empty(batch_shape)
            for idx in np.ndindex(batch_shape):
                eigs = LA.eigvalsh(A[idx])
                result[idx] = sum_largest(eigs, self.k).value
            return result

    def get_data(self):
        """Returns the parameter k.
        """
        return [self.k]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        A = values[0]
        k = self.k
        k_floor = int(np.floor(k))
        k_frac = k - k_floor

        if A.ndim == 2:
            w, v = LA.eigh(A)
            n = A.shape[0]
            # Eigenvalues are sorted ascending, so largest k are the last k.
            D = np.zeros((n, n))
            if k_floor > 0:
                V_top = v[:, -k_floor:]
                D += V_top @ V_top.T
            if k_frac > 0 and k_floor < n:
                v_next = v[:, -(k_floor + 1)]
                D += k_frac * np.outer(v_next, v_next)
            return [sp.csc_array([D.ravel(order='F')]).T]
        else:
            batch_shape = A.shape[:-2]
            n = A.shape[-1]
            total_batch = int(np.prod(batch_shape))
            n2 = n * n
            total_output = total_batch * n2
            rows = []
            cols = []
            vals = []
            for flat_i, idx in enumerate(np.ndindex(batch_shape)):
                mat = A[idx]
                w, v = LA.eigh(mat)
                D = np.zeros((n, n))
                if k_floor > 0:
                    V_top = v[:, -k_floor:]
                    D += V_top @ V_top.T
                if k_frac > 0 and k_floor < n:
                    v_next = v[:, -(k_floor + 1)]
                    D += k_frac * np.outer(v_next, v_next)
                D_flat = D.ravel(order='F')
                for j in range(n2):
                    if D_flat[j] != 0:
                        row = flat_i * n2 + j
                        rows.append(row)
                        cols.append(flat_i)
                        vals.append(D_flat[j])
            grad = sp.csc_array(
                (vals, (rows, cols)), shape=(total_output, total_batch)
            )
            return [grad]

    @property
    def value(self):
        val = self.args[0].value
        if not np.allclose(val, np.swapaxes(val, -2, -1).conj()):
            raise ValueError("Input matrix was not Hermitian/symmetric.")
        if any([p.value is None for p in self.parameters()]):
            return None
        return self._value_impl()
