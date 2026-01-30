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

from typing import List, Tuple

import numpy as np

from cvxpy.expressions.leaf import Leaf


class MatrixSqrt(Leaf):
    """Symbolic matrix square root for deferred factorization.

    MatrixSqrt(P) represents L such that P = L @ L.T (Cholesky-like decomposition).
    The factorization is computed when .value is accessed, using decomp_quad.

    Parameters
    ----------
    P : Expression
        The original P matrix (should be PSD Constant or Parameter expression).
    """

    def __init__(self, P) -> None:
        self._P = P  # The original P matrix
        self._cached_value = None
        self._cached_P_value = None
        n = P.shape[0]
        # Shape is (n, n) as a conservative estimate; actual factorization
        # may have fewer columns if P is rank-deficient
        super(MatrixSqrt, self).__init__((n, n))

    def get_data(self) -> None:
        """Returns data needed to reconstruct the object.

        Note: MatrixSqrt uses _P attribute directly, not get_data.
        """
        return None

    @property
    def value(self):
        """Compute Cholesky-like factorization of P.

        Uses decomp_quad to get scale, M1, M2 where P = scale * (M1 @ M1.T - M2 @ M2.T).
        For convex (PSD) P, M2 is empty, so L = sqrt(scale) * M1.

        Returns
        -------
        ndarray
            L such that P = L @ L.T
        """
        P_val = self._P.value
        if P_val is None:
            return None

        # Cache the factorization if P hasn't changed
        if self._cached_value is not None and self._cached_P_value is not None:
            if np.array_equal(P_val, self._cached_P_value):
                return self._cached_value

        # Import here to avoid circular import
        from cvxpy.atoms.quad_form import decomp_quad
        scale, M1, M2 = decomp_quad(P_val)

        if M2.size > 0:
            raise ValueError(
                "P must be positive semidefinite for SOC conversion. "
                "Got indefinite matrix with negative eigenvalues."
            )

        # Special case: P == 0
        if M1.size == 0:
            n = self._P.shape[0]
            self._cached_value = np.zeros((n, 0))
            self._cached_P_value = P_val.copy() if hasattr(P_val, 'copy') else P_val
            return self._cached_value

        # P = scale * M1 @ M1.T, so L = sqrt(scale) * M1
        L = np.sqrt(scale) * M1

        self._cached_value = L
        self._cached_P_value = P_val.copy() if hasattr(P_val, 'copy') else P_val
        return L

    @value.setter
    def value(self, val):  # noqa: ARG002
        """Cannot set value directly."""
        del val  # unused
        raise ValueError("Cannot set value of MatrixSqrt directly.")

    def name(self) -> str:
        return f"MatrixSqrt({self._P.name()})"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def is_constant(self) -> bool:
        """Is the expression constant (not parametrized)?"""
        return self._P.is_constant()

    def constants(self) -> list:
        """Returns constants in the expression."""
        return self._P.constants()

    def parameters(self) -> List:
        """Returns parameters in the expression."""
        return self._P.parameters()

    def is_nonneg(self) -> bool:
        return False

    def is_nonpos(self) -> bool:
        return False

    def is_imag(self) -> bool:
        return False

    def is_complex(self) -> bool:
        return self._P.is_complex()

    def is_symmetric(self) -> bool:
        return False

    def is_hermitian(self) -> bool:
        return False

    def is_psd(self) -> bool:
        return False

    def is_nsd(self) -> bool:
        return False

    @property
    def grad(self):
        """Gradient is not supported for MatrixSqrt."""
        return {}

    def canonicalize(self):
        """Returns the graph implementation of the object."""
        import cvxpy.lin_ops.lin_utils as lu
        # Create a constant node with the value
        val = self.value
        if val is None:
            raise ValueError("Cannot canonicalize MatrixSqrt with None value")
        obj = lu.create_const(val, val.shape, sparse=False)
        return (obj, [])
