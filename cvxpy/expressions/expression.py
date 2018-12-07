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

from functools import wraps
import warnings

from cvxpy import error
from cvxpy.constraints import Equality, Inequality, PSD
from cvxpy.expressions import cvxtypes
import cvxpy.utilities as u
import cvxpy.utilities.key_utils as ku
import cvxpy.settings as s
import abc
import numpy as np


def _cast_other(binary_op):
    """Casts the second argument of a binary operator as an Expression.

    Args:
        binary_op: A binary operator in the Expression class.

    Returns:
        A wrapped binary operator that can handle non-Expression arguments.
    """

    @wraps(binary_op)
    def cast_op(self, other):
        """A wrapped binary operator that can handle non-Expression arguments.
        """
        other = self.cast_to_const(other)
        return binary_op(self, other)
    return cast_op


class Expression(u.Canonical):
    """A mathematical expression in a convex optimization problem.

    Overloads many operators to allow for convenient creation of compound
    expressions (e.g., the sum of two expressions) and constraints.
    """

    __metaclass__ = abc.ABCMeta

    # Handles arithmetic operator overloading with Numpy.
    __array_priority__ = 100

    @abc.abstractproperty
    def value(self):
        """NumPy.ndarray or None : The numeric value of the expression.
        """
        return NotImplemented

    @abc.abstractproperty
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns
        -------
        dict
            A map of variable to SciPy CSC sparse matrix; None if a variable
            value is missing.
        """
        return NotImplemented

    @abc.abstractproperty
    def domain(self):
        """list : The constraints describing the closure of the region
           where the expression is finite.
        """
        return NotImplemented

    def __str__(self):
        """Returns a string showing the mathematical expression.
        """
        return self.name()

    def __repr__(self):
        """Returns a string with information about the expression.
        """
        return "Expression(%s, %s, %s)" % (self.curvature,
                                           self.sign,
                                           self.shape)

    @abc.abstractmethod
    def name(self):
        """str : The string representation of the expression.
        """
        return NotImplemented

    @property
    def expr(self):
        """Expression : returns itself."""
        return self

    # Curvature properties.

    @property
    def curvature(self):
        """str : The curvature of the expression.
        """
        if self.is_constant():
            curvature_str = s.CONSTANT
        elif self.is_affine():
            curvature_str = s.AFFINE
        elif self.is_convex():
            curvature_str = s.CONVEX
        elif self.is_concave():
            curvature_str = s.CONCAVE
        else:
            curvature_str = s.UNKNOWN
        return curvature_str

    @property
    def log_log_curvature(self):
        """str : The log-log curvature of the expression.
        """
        if self.is_constant():
            curvature_str = s.CONSTANT
        elif self.is_log_log_affine():
            curvature_str = s.LOG_LOG_AFFINE
        elif self.is_log_log_convex():
            curvature_str = s.LOG_LOG_CONVEX
        elif self.is_log_log_concave():
            curvature_str = s.LOG_LOG_CONCAVE
        else:
            curvature_str = s.UNKNOWN
        return curvature_str

    def is_constant(self):
        """Is the expression constant?
        """
        try:
            return self.__is_constant
        except AttributeError:
            self.__is_constant = (len(self.variables()) == 0 or
                                  self.is_zero() or 0 in self.shape)
            return self.__is_constant

    def is_affine(self):
        """Is the expression affine?
        """
        try:
            return self.__is_affine
        except AttributeError:
            self.__is_affine = self.is_constant() or (
                               self.is_convex() and self.is_concave())
            return self.__is_affine

    @abc.abstractmethod
    def is_convex(self):
        """Is the expression convex?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_concave(self):
        """Is the expression concave?
        """
        return NotImplemented

    def is_dcp(self):
        """Checks whether the Expression is DCP.

        Returns
        -------
        bool
            True if the Expression is DCP, False otherwise.
        """
        return self.is_convex() or self.is_concave()

    def is_log_log_affine(self):
        """Is the expression affine?
        """
        try:
            return self.__is_log_log_affine
        except AttributeError:
            self.__is_log_log_affine = (self.is_constant() and self.is_pos()) or (
                                       self.is_log_log_convex() and
                                       self.is_log_log_concave())
            return self.__is_log_log_affine

    @abc.abstractmethod
    def is_log_log_convex(self):
        """Is the expression log-log convex?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_log_log_concave(self):
        """Is the expression log-log concave?
        """
        return NotImplemented

    def is_dgp(self):
        """Checks whether the Expression is log-log DCP.

        Returns
        -------
        bool
            True if the Expression is log-log DCP, False otherwise.
        """
        return self.is_log_log_convex() or self.is_log_log_concave()

    def is_hermitian(self):
        """Is the expression a Hermitian matrix?
        """
        return (self.is_real() and self.is_symmetric())

    def is_psd(self):
        """Is the expression a positive semidefinite matrix?
        """
        # Default to False.
        return False

    def is_nsd(self):
        """Is the expression a negative semidefinite matrix?
        """
        # Default to False.
        return False

    def is_quadratic(self):
        """Is the expression quadratic?
        """
        # Defaults to false
        return False

    def is_symmetric(self):
        """Is the expression symmetric?
        """
        # Defaults to false unless scalar.
        return self.is_scalar()

    def is_pwl(self):
        """Is the expression piecewise linear?
        """
        # Defaults to false
        return False

    def is_qpwa(self):
        """Is the expression quadratic of piecewise affine?
        """
        return self.is_quadratic() or self.is_pwl()

    # Sign properties.

    @property
    def sign(self):
        """str: The sign of the expression.
        """
        if self.is_zero():
            sign_str = s.ZERO
        elif self.is_nonneg():
            sign_str = s.NONNEG
        elif self.is_nonpos():
            sign_str = s.NONPOS
        else:
            sign_str = s.UNKNOWN
        return sign_str

    def is_zero(self):
        """Is the expression all zero?
        """
        try:
            return self.__is_zero
        except AttributeError:
            self.__is_zero = self.is_nonneg() and self.is_nonpos()
            return self.__is_zero

    @abc.abstractmethod
    def is_nonneg(self):
        """Is the expression positive?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_nonpos(self):
        """Is the expression negative?
        """
        return NotImplemented

    def is_pos(self):
        """Is the expression positive and not negative?
        """
        return self.is_nonneg() and not self.is_nonpos()

    def is_neg(self):
        """Is the expression negative and not positive?
        """
        return self.is_nonneg() and not self.is_nonpos()

    @abc.abstractproperty
    def shape(self):
        """tuple : The expression dimensions.
        """
        return NotImplemented

    def is_real(self):
        """Is the Leaf real valued?
        """
        return not self.is_complex()

    @abc.abstractproperty
    def is_imag(self):
        """Is the Leaf imaginary?
        """
        return NotImplemented

    @abc.abstractproperty
    def is_complex(self):
        """Is the Leaf complex valued?
        """
        return NotImplemented

    @property
    def size(self):
        """int : The number of entries in the expression.
        """
        return np.prod(self.shape, dtype=int)

    @property
    def ndim(self):
        """int : The number of dimensions in the expression's shape.
        """
        return len(self.shape)

    def flatten(self):
        """Vectorizes the expression.
        """
        return cvxtypes.vec()(self)

    def is_scalar(self):
        """Is the expression a scalar?
        """
        return all(d == 1 for d in self.shape)

    def is_vector(self):
        """Is the expression a column or row vector?
        """
        return self.ndim <= 1 or (self.ndim == 2 and min(self.shape) == 1)

    def is_matrix(self):
        """Is the expression a matrix?
        """
        return self.ndim == 2 and self.shape[0] > 1 and self.shape[1] > 1

    def __getitem__(self, key):
        """Return a slice/index into the expression.
        """
        # Returning self for scalars causes
        # the built-in sum to hang.
        if isinstance(key, tuple) and len(key) == 0:
            return self
        elif ku.is_special_slice(key):
            return cvxtypes.special_index()(self, key)
        else:
            return cvxtypes.index()(self, key)

    @property
    def T(self):
        """Expression : The transpose of the expression.
        """
        # Transpose of a scalar is that scalar.
        if self.ndim <= 1:
            return self
        else:
            return cvxtypes.transpose()(self)

    @property
    def H(self):
        """Expression : The transpose of the expression.
        """
        if self.is_real():
            return self.T
        else:
            return cvxtypes.conj()(self).T

    def __pow__(self, power):
        """Raise expression to a power.

        Parameters
        ----------
        power : float
            The power to which to raise the expression.

        Returns
        -------
        Expression
            The expression raised to ``power``.
        """
        return cvxtypes.power()(self, power)

    # Arithmetic operators.
    @staticmethod
    def cast_to_const(expr):
        """Converts a non-Expression to a Constant.
        """
        return expr if isinstance(expr, Expression) else cvxtypes.constant()(expr)

    @_cast_other
    def __add__(self, other):
        """Expression : Sum two expressions.
        """
        return cvxtypes.add_expr()([self, other])

    @_cast_other
    def __radd__(self, other):
        """Expression : Sum two expressions.
        """
        return other + self

    @_cast_other
    def __sub__(self, other):
        """Expression : The difference of two expressions.
        """
        return self + -other

    @_cast_other
    def __rsub__(self, other):
        """Expression : The difference of two expressions.
        """
        return other - self

    @_cast_other
    def __mul__(self, other):
        """Expression : The product of two expressions.
        """
        if self.shape == () or other.shape == () or \
           (self.shape[-1] != other.shape[0] and
           (self.is_scalar() or other.is_scalar())):
            return cvxtypes.multiply_expr()(self, other)
        elif self.is_constant() or other.is_constant():
            return cvxtypes.mul_expr()(self, other)
        else:
            if error.warnings_enabled():
                warnings.warn("Forming a nonconvex expression.")
            return cvxtypes.mul_expr()(self, other)

    @_cast_other
    def __matmul__(self, other):
        """Expression : Matrix multiplication of two expressions.
        """
        if self.shape == () or other.shape == ():
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self.__mul__(other)

    @_cast_other
    def __truediv__(self, other):
        """Expression : One expression divided by another.
        """
        return self.__div__(other)

    @_cast_other
    def __div__(self, other):
        """Expression : One expression divided by another.
        """
        # Can only divide by scalar constants.
        if other.is_constant() and other.is_scalar():
            return cvxtypes.div_expr()(self, other)
        elif other.is_scalar():
            if error.warnings_enabled():
                warnings.warn("Forming a nonconvex expression.")
            return cvxtypes.div_expr()(self, other)
        else:
            raise ValueError("Can only divide by a scalar constant, but %s "
                             "is not scalar." % str(other))

    @_cast_other
    def __rdiv__(self, other):
        """Expression : Called for Number / Expression.
        """
        return other / self

    @_cast_other
    def __rtruediv__(self, other):
        """Expression : Called for Number / Expression.
        """
        return other / self

    @_cast_other
    def __rmul__(self, other):
        """Expression : Called for Number * Expression.
        """
        return other * self

    @_cast_other
    def __rmatmul__(self, other):
        """Expression : Called for matrix @ Expression.
        """
        return other.__matmul__(self)

    def __neg__(self):
        """Expression : The negation of the expression.
        """
        return cvxtypes.neg_expr()(self)

    @_cast_other
    def __rshift__(self, other):
        """PSD : Creates a positive semidefinite inequality.
        """
        return PSD(self - other)

    @_cast_other
    def __rrshift__(self, other):
        """PSD : Creates a positive semidefinite inequality.
        """
        return PSD(other - self)

    @_cast_other
    def __lshift__(self, other):
        """PSD : Creates a negative semidefinite inequality.
        """
        return PSD(other - self)

    @_cast_other
    def __rlshift__(self, other):
        """PSD : Creates a negative semidefinite inequality.
        """
        return PSD(self - other)

    # Needed for Python3:
    def __hash__(self):
        return id(self)

    # Comparison operators.
    @_cast_other
    def __eq__(self, other):
        """Equality : Creates a constraint ``self == other``.
        """
        return Equality(self, other)

    @_cast_other
    def __le__(self, other):
        """Inequality : Creates an inequality constraint ``self <= other``.
        """
        return Inequality(self, other)

    def __lt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")

    @_cast_other
    def __ge__(self, other):
        """NonPos : Creates an inequality constraint.
        """
        return other.__le__(self)

    def __gt__(self, other):
        """Unsupported.
        """
        raise NotImplementedError("Strict inequalities are not allowed.")
