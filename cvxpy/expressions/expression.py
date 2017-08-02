"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by

the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings

from cvxpy.constraints import Zero, NonPos, PSD
from cvxpy.error import DCPError
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

    def cast_op(self, other):
        """A wrapped binary operator that can handle non-Expression arguments.
        """
        other = self.cast_to_const(other)
        return binary_op(self, other)
    return cast_op


class Expression(u.Canonical):
    """
    A mathematical expression in a convex optimization problem.
    """

    __metaclass__ = abc.ABCMeta

    # Handles arithmetic operator overloading with Numpy.
    __array_priority__ = 100

    @abc.abstractproperty
    def value(self):
        """Returns the numeric value of the expression.

        Returns:
            A numpy matrix or a scalar.
        """
        return NotImplemented

    @abc.abstractproperty
    def grad(self):
        """Gives the (sub/super)gradient of the expression w.r.t. each variable.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Returns:
            A map of variable to SciPy CSC sparse matrix.
            None if a variable value is missing.
        """
        return NotImplemented

    @abc.abstractproperty
    def domain(self):
        """A list of constraints describing the closure of the region
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
        """Returns the string representation of the expression.
        """
        return NotImplemented

    @property
    def expr(self):
        return self

    # Curvature properties.

    @property
    def curvature(self):
        """ Returns the curvature of the expression.
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

    def is_constant(self):
        """Is the expression constant?
        """
        return len(self.variables()) == 0 or self.is_zero()

    def is_affine(self):
        """Is the expression affine?
        """
        return self.is_constant() or (self.is_convex() and self.is_concave())

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
        """Is the expression DCP compliant? (i.e., no unknown curvatures).
        """
        return self.is_convex() or self.is_concave()

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
        """Returns the sign of the expression.
        """
        if self.is_zero():
            sign_str = s.ZERO
        elif self.is_nonneg():
            sign_str = s.POSITIVE
        elif self.is_nonpos():
            sign_str = s.NEGATIVE
        else:
            sign_str = s.UNKNOWN
        return sign_str

    def is_zero(self):
        """Is the expression all zero?
        """
        return self.is_nonneg() and self.is_nonpos()

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

    @abc.abstractproperty
    def shape(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return NotImplemented

    @property
    def size(self):
        """Returns the number of entries in the expression.
        """
        return np.prod(self.shape)

    @property
    def ndim(self):
        """Returns the number of dimensions.
        """
        return len(self.shape)

    def is_scalar(self):
        """Is the expression a scalar?
        """
        return self.shape == (1, 1)

    def is_vector(self):
        """Is the expression a column or row vector?
        """
        return min(self.shape) == 1

    def is_matrix(self):
        """Is the expression a matrix?
        """
        return self.shape[0] > 1 and self.shape[1] > 1

    def __getitem__(self, key):
        """Return a slice/index into the expression.
        """
        # Returning self for scalars causes
        # the built-in sum to hang.
        if ku.is_special_slice(key):
            return cvxtypes.index().get_special_slice(self, key)
        else:
            return cvxtypes.index()(self, key)

    @property
    def T(self):
        """The transpose of an expression.
        """
        # Transpose of a scalar is that scalar.
        if self.is_scalar():
            return self
        else:
            return cvxtypes.transpose()(self)

    def __pow__(self, power):
        """The power operator.
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
        """The sum of two expressions.
        """
        return cvxtypes.add_expr()([self, other])

    @_cast_other
    def __radd__(self, other):
        """Called for Number + Expression.
        """
        return other + self

    @_cast_other
    def __sub__(self, other):
        """The difference of two expressions.
        """
        return self + -other

    @_cast_other
    def __rsub__(self, other):
        """Called for Number - Expression.
        """
        return other - self

    @_cast_other
    def __mul__(self, other):
        """The product of two expressions.
        """
        # Multiplying by a constant on the right is handled differently
        # from multiplying by a constant on the left.
        if self.is_constant():
            # TODO HACK catch c.T*x where c is a NumPy 1D array.
            # TODO(akshayka): This logic will need to change once in order to
            # handle 1D and 0D arrays.
            if self.shape[0] == other.shape[0] and \
               self.shape[1] != self.shape[0] and \
               isinstance(self, cvxtypes.constant()) and self.is_1D_array:
                self = self.T

        if self.is_constant() or other.is_constant():
            if other.is_scalar() and self.shape[1] != 1:
                lh_arg = cvxtypes.reshape()(self, (self.size, 1))
                prod = cvxtypes.mul_expr()(lh_arg, other)
                return cvxtypes.reshape()(prod, self.shape)
            elif self.is_scalar() and other.shape[0] != 1:
                lh_arg = cvxtypes.reshape()(other, (other.size, 1))
                prod = cvxtypes.mul_expr()(lh_arg, self)
                return cvxtypes.reshape()(prod, other.shape)
            else:
                return cvxtypes.mul_expr()(self, other)
        else:
            warnings.warn("Forming a nonconvex expression.")
            return cvxtypes.mul_expr()(self, other)

    @_cast_other
    def __matmul__(self, other):
        """Matrix multiplication of two expressions.
        """
        if self.is_scalar() or other.is_scalar():
            raise ValueError("Scalar operands are not allowed, use '*' instead")
        return self.__mul__(other)

    @_cast_other
    def __truediv__(self, other):
        """One expression divided by another.
        """
        return self.__div__(other)

    @_cast_other
    def __div__(self, other):
        """One expression divided by another.
        """
        # Can only divide by scalar constants.
        if other.is_constant() and other.is_scalar():
            return cvxtypes.div_expr()(self, other)
        else:
            raise DCPError("Can only divide by a scalar constant.")

    @_cast_other
    def __rdiv__(self, other):
        """Called for Number / Expression.
        """
        return other / self

    @_cast_other
    def __rtruediv__(self, other):
        """Called for Number / Expression.
        """
        return other / self

    @_cast_other
    def __rmul__(self, other):
        """Called for Number * Expression.
        """
        return other * self

    @_cast_other
    def __rmatmul__(self, other):
        """Called for matrix @ Expression.
        """
        return other.__matmul__(self)

    def __neg__(self):
        """The negation of the expression.
        """
        return cvxtypes.neg_expr()(self)

    @_cast_other
    def __rshift__(self, other):
        """Positive definite inequality.
        """
        return PSD(self - other)

    @_cast_other
    def __rrshift__(self, other):
        """Positive definite inequality.
        """
        return PSD(other - self)

    @_cast_other
    def __lshift__(self, other):
        """Positive definite inequality.
        """
        return PSD(other - self)

    @_cast_other
    def __rlshift__(self, other):
        """Positive definite inequality.
        """
        return PSD(self - other)

    # Needed for Python3:
    def __hash__(self):
        return id(self)

    # Comparison operators.
    @_cast_other
    def __eq__(self, other):
        """Returns an equality constraint.
        """
        return Zero(self - other)

    @_cast_other
    def __le__(self, other):
        """Returns an inequality constraint.
        """
        return NonPos(self - other)

    def __lt__(self, other):
        """Returns an inequality constraint.
        """
        return self <= other

    @_cast_other
    def __ge__(self, other):
        """Returns an inequality constraint.
        """
        return other.__le__(self)

    def __gt__(self, other):
        """Returns an inequality constraint.
        """
        return self >= other
