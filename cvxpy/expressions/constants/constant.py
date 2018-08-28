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

from fastcache import clru_cache
import cvxpy.interface as intf
from cvxpy.expressions.leaf import Leaf
from cvxpy.settings import EIGVAL_TOL
import cvxpy.lin_ops.lin_utils as lu
from scipy import linalg as LA
import numpy as np


class Constant(Leaf):
    """
    A constant value.

    Raw numerical constants (Python primite types, NumPy ndarrays,
    and NumPy matrices) are implicitly cast to constants via Expression
    operator overloading. For example, if ``x`` is an expression and
    ``c`` is a raw constant, then ``x + c`` creates an expression by
    casting ``c`` to a Constant.
    """

    def __init__(self, value):
        # Keep sparse matrices sparse.
        if intf.is_sparse(value):
            self._value = intf.DEFAULT_SPARSE_INTF.const_to_matrix(
                value, convert_scalars=True)
            self._sparse = True
        else:
            self._value = intf.DEFAULT_INTF.const_to_matrix(value)
            self._sparse = False
        self._imag = None
        self._nonneg = self._nonpos = None
        self._symm = None
        self._herm = None
        self._eigvals = None
        super(Constant, self).__init__(intf.shape(self.value))

    def name(self):
        """The value as a string.
        """
        return str(self.value)

    def constants(self):
        """Returns self as a constant.
        """
        return [self]

    @property
    def value(self):
        """NumPy.ndarray or None: The numeric value of the constant.
        """
        return self._value

    @property
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix or None.
        """
        return {}

    @property
    def shape(self):
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

    def __repr__(self):
        """Returns a string with information about the expression.
        """
        return "Constant(%s, %s, %s)" % (self.curvature,
                                         self.sign,
                                         self.shape)

    def is_nonneg(self):
        """Is the expression nonnegative?
        """
        if self._nonneg is None:
            self._compute_attr()
        return self._nonneg

    def is_nonpos(self):
        """Is the expression nonpositive?
        """
        if self._nonpos is None:
            self._compute_attr()
        return self._nonpos

    def is_imag(self):
        """Is the Leaf imaginary?
        """
        if self._imag is None:
            self._compute_attr()
        return self._imag

    @clru_cache(maxsize=100)
    def is_complex(self):
        """Is the Leaf complex valued?
        """
        return np.iscomplexobj(self.value)

    @clru_cache(maxsize=100)
    def is_symmetric(self):
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

    @clru_cache(maxsize=100)
    def is_hermitian(self):
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

    def _compute_attr(self):
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

    def _compute_symm_attr(self):
        """Determine whether the constant is symmetric/Hermitian.
        """
        # Set DCP attributes.
        is_symm, is_herm = intf.is_hermitian(self.value)
        self._symm = is_symm
        self._herm = is_herm

    @clru_cache(maxsize=100)
    def is_psd(self):
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

        # Compute eigenvalues if absent.
        if self._eigvals is None:
            self._compute_eigvals()
        return all(self._eigvals.real >= -EIGVAL_TOL)

    @clru_cache(maxsize=100)
    def is_nsd(self):
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

        # Compute eigenvalues if absent.
        if self._eigvals is None:
            self._compute_eigvals()
        return all(self._eigvals.real <= EIGVAL_TOL)

    def _compute_eigvals(self):
        """Compute the eigenvalues of the Hermitian or
        symmetric matrix represented by this constant.
        """
        if self._sparse:
            self._eigvals = LA.eigvalsh(self.value.todense())
        else:
            self._eigvals = LA.eigvalsh(self.value)
