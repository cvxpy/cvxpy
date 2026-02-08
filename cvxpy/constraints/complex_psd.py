"""
Copyright, the CVXPY authors

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

from cvxpy.constraints.cones import Cone
from cvxpy.utilities import scopes


class ComplexPSD(Cone):
    """Hermitian positive semidefinite cone: R + i*I >= 0.

    Two args: [real_part (n x n), imag_part (n x n)].
    real_part should be symmetric, imag_part should be skew-symmetric,
    so that R + i*I is Hermitian.

    Parameters
    ----------
    real_part : Expression
        The real part of the Hermitian matrix (n x n).
    imag_part : Expression
        The imaginary part of the Hermitian matrix (n x n).
    constr_id : int
        A unique id for the constraint.
    """

    def __init__(self, real_part, imag_part, constr_id=None) -> None:
        if len(real_part.shape) != 2 or real_part.shape[0] != real_part.shape[1]:
            raise ValueError(
                "Non-square matrix in ComplexPSD constraint (real part)."
            )
        if len(imag_part.shape) != 2 or imag_part.shape[0] != imag_part.shape[1]:
            raise ValueError(
                "Non-square matrix in ComplexPSD constraint (imag part)."
            )
        if real_part.shape != imag_part.shape:
            raise ValueError(
                "Real and imaginary parts must have the same shape "
                "in ComplexPSD constraint."
            )
        super(ComplexPSD, self).__init__([real_part, imag_part], constr_id)

    def name(self) -> str:
        return "ComplexPSD(%s, %s)" % (self.args[0], self.args[1])

    @property
    def shape(self):
        return self.args[0].shape

    @property
    def size(self):
        n = self.args[0].shape[0]
        return n * n

    def is_dcp(self, dpp: bool = False) -> bool:
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine() and self.args[1].is_affine()
        return self.args[0].is_affine() and self.args[1].is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def residual(self):
        if self.args[0].value is None or self.args[1].value is None:
            return None
        H = self.args[0].value + 1j * self.args[1].value
        eigs = np.linalg.eigvalsh(H)
        return np.maximum(-np.min(eigs), 0)
