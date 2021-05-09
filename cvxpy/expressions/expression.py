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
from cvxpy.utilities import scopes
import cvxpy.utilities.performance_utils as perf
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


__STAR_MATMUL_COUNT__ = 1

__STAR_MATMUL_WARNING__ = """
This use of ``*`` has resulted in matrix multiplication.
Using ``*`` for matrix multiplication has been deprecated since CVXPY 1.1.
    Use ``*`` for matrix-scalar and vector-scalar multiplication.
    Use ``@`` for matrix-matrix and matrix-vector multiplication.
    Use ``multiply`` for elementwise multiplication.
This code path has been hit %s times so far.
"""


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
        raise NotImplementedError()

    def _value_impl(self):
        """Implementation of .value.
        """
        return self.value

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
        raise NotImplementedError()

    @abc.abstractproperty
    def domain(self):
        """list : The constraints describing the closure of the region
           where the expression is finite.
        """
        raise NotImplementedError()

    def __str__(self):
        """Returns a string showing the mathematical expression.
        """
        return self.name()

    def __repr__(self) -> str:
        """Returns a string with information about the expression.
        """
        return "Expression(%s, %s, %s)" % (self.curvature,
                                           self.sign,
                                           self.shape)

    @abc.abstractmethod
    def name(self):
        """str : The string representation of the expression.
        """
        raise NotImplementedError()

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
        elif self.is_quasilinear():
            curvature_str = s.QUASILINEAR
        elif self.is_quasiconvex():
            curvature_str = s.QUASICONVEX
        elif self.is_quasiconcave():
            curvature_str = s.QUASICONCAVE
        else:
            curvature_str = s.UNKNOWN
        return curvature_str

    @property
    def log_log_curvature(self):
        """str : The log-log curvature of the expression.
        """
        if self.is_log_log_constant():
            curvature_str = s.LOG_LOG_CONSTANT
        elif self.is_log_log_affine():
            curvature_str = s.LOG_LOG_AFFINE
        elif self.is_log_log_convex():
            curvature_str = s.LOG_LOG_CONVEX
        elif self.is_log_log_concave():
            curvature_str = s.LOG_LOG_CONCAVE
        else:
            curvature_str = s.UNKNOWN
        return curvature_str

    @perf.compute_once
    def is_constant(self) -> bool:
        """Is the expression constant?
        """
        return 0 in self.shape or all(
            arg.is_constant() for arg in self.args)

    @perf.compute_once
    def is_affine(self) -> bool:
        """Is the expression affine?
        """
        return self.is_constant() or (self.is_convex() and self.is_concave())

    @abc.abstractmethod
    def is_convex(self) -> bool:
        """Is the expression convex?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_concave(self) -> bool:
        """Is the expression concave?
        """
        raise NotImplementedError()

    @perf.compute_once
    def is_dcp(self, dpp: bool = False) -> bool:
        """Checks whether the Expression is DCP.

        Parameters
        ----------
        dpp : bool, optional
            If True, enforce the disciplined parametrized programming (DPP)
            ruleset; only relevant when the problem involves Parameters.

        Returns
        -------
        bool
            True if the Expression is DCP, False otherwise.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.is_convex() or self.is_concave()
        return self.is_convex() or self.is_concave()

    def is_log_log_constant(self) -> bool:
        """Is the expression log-log constant, ie, elementwise positive?
        """
        if not self.is_constant():
            return False

        if isinstance(self, (cvxtypes.constant(), cvxtypes.parameter())):
            return self.is_pos()
        else:
            return self.value is not None and np.all(self.value > 0)

    @perf.compute_once
    def is_log_log_affine(self) -> bool:
        """Is the expression affine?
        """
        return (self.is_log_log_constant()
                or (self.is_log_log_convex() and self.is_log_log_concave()))

    @abc.abstractmethod
    def is_log_log_convex(self) -> bool:
        """Is the expression log-log convex?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_log_log_concave(self) -> bool:
        """Is the expression log-log concave?
        """
        raise NotImplementedError()

    def is_dgp(self, dpp: bool = False) -> bool:
        """Checks whether the Expression is log-log DCP.

        Returns
        -------
        bool
            True if the Expression is log-log DCP, False otherwise.
        """
        if dpp:
            with scopes.dpp_scope():
                return self.is_log_log_convex() or self.is_log_log_concave()
        return self.is_log_log_convex() or self.is_log_log_concave()

    @abc.abstractmethod
    def is_dpp(self, context='dcp') -> bool:
        """The expression is a disciplined parameterized expression.
        """
        raise NotImplementedError()

    def is_quasiconvex(self) -> bool:
        return self.is_convex()

    def is_quasiconcave(self) -> bool:
        return self.is_concave()

    def is_quasilinear(self) -> bool:
        return self.is_quasiconvex() and self.is_quasiconcave()

    @perf.compute_once
    def is_dqcp(self) -> bool:
        """Checks whether the Expression is DQCP.

        Returns
        -------
        bool
            True if the Expression is DQCP, False otherwise.
        """
        return self.is_quasiconvex() or self.is_quasiconcave()

    def is_hermitian(self) -> bool:
        """Is the expression a Hermitian matrix?
        """
        return (self.is_real() and self.is_symmetric())

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        # Default to False.
        return False

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        # Default to False.
        return False

    def is_quadratic(self) -> bool:
        """Is the expression quadratic?
        """
        # Defaults to is constant.
        return self.is_constant()

    def is_symmetric(self) -> bool:
        """Is the expression symmetric?
        """
        # Defaults to false unless scalar.
        return self.is_scalar()

    def is_pwl(self) -> bool:
        """Is the expression piecewise linear?
        """
        # Defaults to constant.
        return self.is_constant()

    def is_qpwa(self) -> bool:
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

    @perf.compute_once
    def is_zero(self) -> bool:
        """Is the expression all zero?
        """
        return self.is_nonneg() and self.is_nonpos()

    @abc.abstractmethod
    def is_nonneg(self) -> bool:
        """Is the expression positive?
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_nonpos(self) -> bool:
        """Is the expression negative?
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def shape(self):
        """tuple : The expression dimensions.
        """
        raise NotImplementedError()

    def is_real(self) -> bool:
        """Is the Leaf real valued?
        """
        return not self.is_complex()

    @abc.abstractproperty
    def is_imag(self) -> bool:
        """Is the Leaf imaginary?
        """
        raise NotImplementedError()

    @abc.abstractproperty
    def is_complex(self) -> bool:
        """Is the Leaf complex valued?
        """
        raise NotImplementedError()

    @property
    def size(self):
        """int : The number of entries in the expression.
        """
        return np.prod(self.shape, dtype=int)

    @property
    def ndim(self) -> int:
        """int : The number of dimensions in the expression's shape.
        """
        return len(self.shape)

    def flatten(self):
        """Vectorizes the expression.
        """
        return cvxtypes.vec()(self)

    def is_scalar(self) -> bool:
        """Is the expression a scalar?
        """
        return all(d == 1 for d in self.shape)

    def is_vector(self) -> bool:
        """Is the expression a column or row vector?
        """
        return self.ndim <= 1 or (self.ndim == 2 and min(self.shape) == 1)

    def is_matrix(self) -> bool:
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
        if isinstance(expr, list):
            for elem in expr:
                if isinstance(elem, Expression):
                    raise ValueError(
                        "The input must be a single CVXPY Expression, not a list. "
                        "Combine Expressions using atoms such as bmat, hstack, and vstack."
                    )
        return expr if isinstance(expr, Expression) else cvxtypes.constant()(expr)

    @staticmethod
    def broadcast(lh_expr, rh_expr):
        """Broacast the binary operator.
        """
        lh_expr = Expression.cast_to_const(lh_expr)
        rh_expr = Expression.cast_to_const(rh_expr)
        if lh_expr.is_scalar() and not rh_expr.is_scalar():
            lh_expr = cvxtypes.promote()(lh_expr, rh_expr.shape)
        elif rh_expr.is_scalar() and not lh_expr.is_scalar():
            rh_expr = cvxtypes.promote()(rh_expr, lh_expr.shape)
        # Broadcasting.
        if lh_expr.ndim == 2 and rh_expr.ndim == 2:
            # Replicate dimensions of size 1.
            dims = [max(lh_expr.shape[i], rh_expr.shape[i]) for i in range(2)]
            # Broadcast along dim 0.
            if lh_expr.shape[0] == 1 and lh_expr.shape[0] < dims[0]:
                lh_expr = np.ones((dims[0], 1)) @ lh_expr
            if rh_expr.shape[0] == 1 and rh_expr.shape[0] < dims[0]:
                rh_expr = np.ones((dims[0], 1)) @ rh_expr
            # Broadcast along dim 1.
            if lh_expr.shape[1] == 1 and lh_expr.shape[1] < dims[1]:
                lh_expr = lh_expr @ np.ones((1, dims[1]))
            if rh_expr.shape[1] == 1 and rh_expr.shape[1] < dims[1]:
                rh_expr = rh_expr @ np.ones((1, dims[1]))
        return lh_expr, rh_expr

    @_cast_other
    def __add__(self, other):
        """Expression : Sum two expressions.
        """
        if isinstance(other, cvxtypes.constant()) and other.is_zero():
            return self
        self, other = self.broadcast(self, other)
        return cvxtypes.add_expr()([self, other])

    @_cast_other
    def __radd__(self, other):
        """Expression : Sum two expressions.
        """
        if isinstance(other, cvxtypes.constant()) and other.is_zero():
            return self
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
        if self.shape == () or other.shape == ():
            # Use one argument to apply a scaling to the remaining argument.
            # We accomplish this with elementwise multiplication, which
            # casts the scalar argument to match the size of the remaining
            # argument.
            return cvxtypes.elmul_expr()(self, other)
        elif self.shape[-1] != other.shape[0] and \
                (self.is_scalar() or other.is_scalar()):
            # If matmul was intended, this gives a dimension mismatch. We
            # interpret the ``is_scalar`` results as implying that the user
            # simply wants to apply a scaling.
            return cvxtypes.elmul_expr()(self, other)
        else:
            # The only reasonable interpretation is that the user intends
            # to apply matmul. There might be a dimension mismatch, but we
            # don't check for that here.
            if not (self.is_constant() or other.is_constant()):
                if error.warnings_enabled():
                    warnings.warn("Forming a nonconvex expression.")
            # Because we want to discourage using ``*`` to call matmul, we
            # raise a warning to the user.
            with warnings.catch_warnings():
                global __STAR_MATMUL_COUNT__
                warnings.simplefilter("always", UserWarning, append=True)
                msg = __STAR_MATMUL_WARNING__ % __STAR_MATMUL_COUNT__
                warnings.warn(msg, UserWarning)
                warnings.warn(msg, DeprecationWarning)
                __STAR_MATMUL_COUNT__ += 1
            return cvxtypes.matmul_expr()(self, other)

    @_cast_other
    def __matmul__(self, other):
        """Expression : Matrix multiplication of two expressions.
        """
        if self.shape == () or other.shape == ():
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return cvxtypes.matmul_expr()(self, other)

    @_cast_other
    def __truediv__(self, other):
        """Expression : One expression divided by another.
        """
        return self.__div__(other)

    @_cast_other
    def __div__(self, other):
        """Expression : One expression divided by another.
        """
        self, other = self.broadcast(self, other)
        if (self.is_scalar() or other.is_scalar()) or other.shape == self.shape:
            return cvxtypes.div_expr()(self, other)
        else:
            raise ValueError("Incompatible shapes for division (%s / %s)" % (
                             self.shape, other.shape))

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
        if self.shape == () or other.shape == ():
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return cvxtypes.matmul_expr()(other, self)

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
    def __hash__(self) -> int:
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
