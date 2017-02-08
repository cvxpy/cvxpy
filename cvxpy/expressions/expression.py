"""
Copyright 2017 Steven Diamond

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

from cvxpy.error import DCPError
import warnings
import cvxpy.utilities as u
import cvxpy.utilities.key_utils as ku
import cvxpy.settings as s
from cvxpy.constraints import EqConstraint, LeqConstraint, PSDConstraint
from cvxpy.expressions import cvxtypes
import abc


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
                                           self.size)

    @abc.abstractmethod
    def name(self):
        """Returns the string representation of the expression.
        """
        return NotImplemented

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

    def is_pwl(self):
        """Is the expression piecewise linear?
        """
        # Defaults to false
        return False

    # Sign properties.

    @property
    def sign(self):
        """Returns the sign of the expression.
        """
        if self.is_zero():
            sign_str = s.ZERO
        elif self.is_positive():
            sign_str = s.POSITIVE
        elif self.is_negative():
            sign_str = s.NEGATIVE
        else:
            sign_str = s.UNKNOWN
        return sign_str

    def is_zero(self):
        """Is the expression all zero?
        """
        return self.is_positive() and self.is_negative()

    @abc.abstractmethod
    def is_positive(self):
        """Is the expression positive?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_negative(self):
        """Is the expression negative?
        """
        return NotImplemented

    @abc.abstractproperty
    def size(self):
        """Returns the (row, col) dimensions of the expression.
        """
        return NotImplemented

    def is_scalar(self):
        """Is the expression a scalar?
        """
        return self.size == (1, 1)

    def is_vector(self):
        """Is the expression a column or row vector?
        """
        return min(self.size) == 1

    def is_matrix(self):
        """Is the expression a matrix?
        """
        return self.size[0] > 1 and self.size[1] > 1

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
            if self.size[0] == other.size[0] and \
               self.size[1] != self.size[0] and \
               isinstance(self, cvxtypes.constant()) and self.is_1D_array:
                self = self.T
            return cvxtypes.mul_expr()(self, other)
        elif other.is_constant():
            # Having the constant on the left is more efficient.
            if self.is_scalar() or other.is_scalar():
                return cvxtypes.mul_expr()(other, self)
            else:
                return cvxtypes.rmul_expr()(self, other)
        # When both expressions are not constant
        # Allow affine * affine but raise DCPError otherwise
        # Cannot multiply two non-constant expressions.
        elif self.is_affine() and other.is_affine():
            warnings.warn("Forming a nonconvex expression (affine)*(affine).")
            return cvxtypes.affine_prod_expr()(self, other)
        else:
            raise DCPError("Cannot multiply %s and %s." % (self.curvature, other.curvature))

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
        return PSDConstraint(self, other)

    @_cast_other
    def __rrshift__(self, other):
        """Positive definite inequality.
        """
        return PSDConstraint(other, self)

    @_cast_other
    def __lshift__(self, other):
        """Positive definite inequality.
        """
        return PSDConstraint(other, self)

    @_cast_other
    def __rlshift__(self, other):
        """Positive definite inequality.
        """
        return PSDConstraint(self, other)

    # Needed for Python3:
    def __hash__(self):
        return id(self)

    # Comparison operators.
    @_cast_other
    def __eq__(self, other):
        """Returns an equality constraint.
        """
        return EqConstraint(self, other)

    @_cast_other
    def __le__(self, other):
        """Returns an inequality constraint.
        """
        return LeqConstraint(self, other)

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
