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

from typing import List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities.eigvals as eig_util
from cvxpy.expressions.leaf import Leaf
from cvxpy.settings import EIGVAL_TOL
from cvxpy.utilities import performance_utils as perf


class Constant(Leaf):
    """
    A constant value.

    Raw numerical constants (Python primite types, NumPy ndarrays,
    and NumPy matrices) are implicitly cast to constants via Expression
    operator overloading. For example, if ``x`` is an expression and
    ``c`` is a raw constant, then ``x + c`` creates an expression by
    casting ``c`` to a Constant.
    """

    def __init__(self, value) -> None:
        # Keep sparse matrices sparse.
        if intf.is_sparse(value):
            self._value = intf.DEFAULT_SPARSE_INTF.const_to_matrix(
                value, convert_scalars=True)
            self._sparse = True
        else:
            self._value = intf.DEFAULT_INTF.const_to_matrix(value)
            self._sparse = False
        self._imag: Optional[bool] = None
        self._nonneg: Optional[bool] = None
        self._nonpos: Optional[bool] = None
        self._symm: Optional[bool] = None
        self._herm: Optional[bool] = None
        self._psd_test: Optional[bool] = None
        self._nsd_test: Optional[bool] = None
        self._cached_is_pos = None
        super(Constant, self).__init__(intf.shape(self.value))

    def name(self) -> str:
        """The value as a string.
        """
        return str(self.value)

    def constants(self) -> List["Constant"]:
        """Returns self as a constant.
        """
        return [self]

    def is_constant(self) -> bool:
        return True

    @property
    def value(self):
        """NumPy.ndarray or None: The numeric value of the constant.
        """
        return self._value

    def is_pos(self) -> bool:
        """Returns whether the constant is elementwise positive.
        """
        if self._cached_is_pos is None:
            if sp.issparse(self._value):
                # sparse constants cannot be elementwise positive,
                # since they (typically) have many entries which are zero.
                self._cached_is_pos = False
            else:
                self._cached_is_pos = np.all(self._value > 0)
        return self._cached_is_pos

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {}

    @property
    def shape(self) -> Tuple[int, ...]:
        """Returns the (row, col) dimensions of the expression.
        """
        return self._shape

    def canonicalize(self):
        """Returns the graph implementation of the object.

        Returns:
            A tuple of (affine expression, [constraints]).
        """
        obj = lu.create_const(self.value, self.shape, self._sparse)
        return (obj, [])

    def __repr__(self) -> str:
        """Returns a string with information about the expression.
        """
        return "Constant(%s, %s, %s)" % (self.curvature,
                                         self.sign,
                                         self.shape)

    def is_nonneg(self) -> bool:
        """Is the expression nonnegative?
        """
        if self._nonneg is None:
            self._compute_attr()
        return self._nonneg

    def is_nonpos(self) -> bool:
        """Is the expression nonpositive?
        """
        if self._nonpos is None:
            self._compute_attr()
        return self._nonpos

    def is_imag(self) -> bool:
        """Is the Leaf imaginary?
        """
        if self._imag is None:
            self._compute_attr()
        return self._imag

    @perf.compute_once
    def is_complex(self) -> bool:
        """Is the Leaf complex valued?
        """
        return np.iscomplexobj(self.value)

    @perf.compute_once
    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        if self.is_scalar():
            return True
        elif self.ndim == 2 and self.shape[0] == self.shape[1]:
            if self._symm is None:
                self._compute_symm_attr()
            return self._symm
        else:
            return False

    @perf.compute_once
    def is_hermitian(self) -> bool:
        """Is the expression a Hermitian matrix?
        """
        if self.is_scalar() and self.is_real():
            return True
        elif self.ndim == 2 and self.shape[0] == self.shape[1]:
            if self._herm is None:
                self._compute_symm_attr()
            return self._herm
        else:
            return False

    def _compute_attr(self) -> None:
        """Compute the attributes of the constant related to complex/real, sign.
        """
        # Set DCP attributes.
        is_real, is_imag = intf.is_complex(self.value)
        if self.is_complex():
            is_nonneg = is_nonpos = False
        else:
            is_nonneg, is_nonpos = intf.sign(self.value)
        self._imag = (is_imag and not is_real)
        self._nonpos = is_nonpos
        self._nonneg = is_nonneg

    def _compute_symm_attr(self) -> None:
        """Determine whether the constant is symmetric/Hermitian.
        """
        # Set DCP attributes.
        is_symm, is_herm = intf.is_hermitian(self.value)
        self._symm = is_symm
        self._herm = is_herm

    @perf.compute_once
    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        # Symbolic only cases.
        if self.is_scalar() and self.is_nonneg():
            return True
        elif self.is_scalar():
            return False
        elif self.ndim == 1:
            return False
        elif self.ndim == 2 and self.shape[0] != self.shape[1]:
            return False
        elif not self.is_hermitian():
            return False

        # Compute sign of bottom eigenvalue if absent.
        if self._psd_test is None:
            self._psd_test = eig_util.is_psd_within_tol(self.value, EIGVAL_TOL)

        return self._psd_test

    @perf.compute_once
    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        # Symbolic only cases.
        if self.is_scalar() and self.is_nonpos():
            return True
        elif self.is_scalar():
            return False
        elif self.ndim == 1:
            return False
        elif self.ndim == 2 and self.shape[0] != self.shape[1]:
            return False
        elif not self.is_hermitian():
            return False

        # Compute sign of top eigenvalue if absent.
        if self._nsd_test is None:
            self._nsd_test = eig_util.is_psd_within_tol(-self.value, EIGVAL_TOL)

        return self._nsd_test
