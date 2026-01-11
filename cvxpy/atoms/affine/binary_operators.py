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

import operator as op
from functools import reduce
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.lin_ops.lin_op as lo
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.broadcast_to import broadcast_to
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.reshape import deep_flatten, reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DCPError
from cvxpy.expressions.constants.parameter import (
    is_param_affine,
    is_param_free,
)
from cvxpy.expressions.expression import Expression


class BinaryOperator(AffAtom):
    """
    Base class for expressions involving binary operators. (other than addition)

    """

    OP_NAME = 'BINARY_OP'

    def __init__(self, lh_exp, rh_exp) -> None:
        super(BinaryOperator, self).__init__(lh_exp, rh_exp)

    def name(self):
        pretty_args = []
        for i, a in enumerate(self.args):
            # Always parenthesize AddExpression and DivExpression
            if isinstance(a, (AddExpression, DivExpression)):
                pretty_args.append('(' + a.name() + ')')
            # For division, also parenthesize multiplication on the right
            elif isinstance(self, DivExpression) and i == 1 and \
                    isinstance(a, (MulExpression, multiply)):
                pretty_args.append('(' + a.name() + ')')
            else:
                pretty_args.append(a.name())
        return pretty_args[0] + ' ' + self.OP_NAME + ' ' + pretty_args[1]
    
    def format_labeled(self):
        """Format binary operation with labels where available."""
        # Check for own label first
        if self._label is not None:
            return self._label
        
        # Build from sub-expressions using their labels
        pretty_args = []
        for i, a in enumerate(self.args):
            # Always parenthesize AddExpression and DivExpression
            if isinstance(a, (AddExpression, DivExpression)):
                pretty_args.append('(' + a.format_labeled() + ')')
            # For division, also parenthesize multiplication on the right
            elif isinstance(self, DivExpression) and i == 1 and \
                    isinstance(a, (MulExpression, multiply)):
                pretty_args.append('(' + a.format_labeled() + ')')
            else:
                pretty_args.append(a.format_labeled())
        return pretty_args[0] + ' ' + self.OP_NAME + ' ' + pretty_args[1]

    def numeric(self, values):
        """Applies the binary operator to the values.
        """
        return reduce(self.OP_FUNC, values)

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Default to rules for times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_imag(self) -> bool:
        """Is the expression imaginary?
        """
        return (self.args[0].is_imag() and self.args[1].is_real()) or \
            (self.args[0].is_real() and self.args[1].is_imag())

    def is_complex(self) -> bool:
        """Is the expression complex valued?
        """
        return (self.args[0].is_complex() or self.args[1].is_complex()) and \
            not (self.args[0].is_imag() and self.args[1].is_imag())


def matmul(lh_exp, rh_exp) -> "MulExpression":
    """Matrix multiplication."""
    return MulExpression(lh_exp, rh_exp)


class MulExpression(BinaryOperator):
    """Matrix multiplication.

    The semantics of multiplication are exactly as those of NumPy's
    matmul function, except here multiplication by a scalar is permitted.
    MulExpression objects can be created by using the '*' operator of
    the Expression class.

    Parameters
    ----------
    lh_exp : Expression
        The left-hand side of the multiplication.
    rh_exp : Expression
        The right-hand side of the multiplication.
    """

    OP_NAME = "@"
    OP_FUNC = op.mul

    def __init__(self, lh_exp, rh_exp) -> None:
        # Broadcast batch dimensions for ND matmul
        lh_exp, rh_exp = self._broadcast_batch_dims(lh_exp, rh_exp)
        super(MulExpression, self).__init__(lh_exp, rh_exp)

    @staticmethod
    def _broadcast_batch_dims(lh_exp, rh_exp):
        """
        Broadcast batch dimensions for ND matrix multiplication.

        For A @ B where A has shape (...a, m, k) and B has shape (...b, k, n),
        broadcasts both to have batch shape broadcast(...a, ...b).
        """
        lh_exp = Expression.cast_to_const(lh_exp)
        rh_exp = Expression.cast_to_const(rh_exp)

        lh_shape = lh_exp.shape
        rh_shape = rh_exp.shape

        # Only apply batch broadcasting for ND arrays (ndim > 2)
        if len(lh_shape) <= 2 and len(rh_shape) <= 2:
            return lh_exp, rh_exp

        # Extract batch dimensions (all but last 2)
        lh_batch = lh_shape[:-2] if len(lh_shape) > 2 else ()
        rh_batch = rh_shape[:-2] if len(rh_shape) > 2 else ()

        # Compute broadcast batch shape
        try:
            broadcast_batch = np.broadcast_shapes(lh_batch, rh_batch)
        except ValueError:
            # Let shape validation handle the error with a clearer message
            return lh_exp, rh_exp

        # Broadcast lhs if needed
        if lh_batch != broadcast_batch:
            target_shape = broadcast_batch + lh_shape[-2:]
            lh_exp = broadcast_to(lh_exp, target_shape)

        # Broadcast rhs if needed
        if rh_batch != broadcast_batch:
            target_shape = broadcast_batch + rh_shape[-2:]
            rh_exp = broadcast_to(rh_exp, target_shape)

        return lh_exp, rh_exp

    def numeric(self, values):
        """Matrix multiplication.
        """
        if values[0].shape == () or values[1].shape == ():
            return values[0] * values[1]
        else:
            return values[0] @ values[1]

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.mul_shapes(self.args[0].shape, self.args[1].shape)

    def is_atom_convex(self) -> bool:
        """Multiplication is convex (affine) in its arguments only if one of
           the arguments is constant.
        """
        if u.scopes.dpp_scope_active():
            # This branch applies curvature rules for DPP.
            #
            # Because a DPP scope is active, parameters will be
            # treated as affine (like variables, not constants) by curvature
            # analysis methods.
            #
            # Like under DCP, a product x * y is convex if x or y is constant.
            # If neither x nor y is constant, then the product is DPP
            # if one of the expressions is affine in its parameters and the
            # other is parameter-free.
            x = self.args[0]
            y = self.args[1]
            return ((x.is_constant() or y.is_constant()) or
                    (is_param_affine(x) and is_param_free(y)) or
                    (is_param_affine(y) and is_param_free(x)))
        else:
            return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        """If the multiplication atom is convex, then it is affine.
        """
        return self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1-idx].is_nonneg()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        return self.args[1-idx].is_nonpos()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.
        CVXPY convention: grad[i, j] = d(output[j]) / d(input[i])
        Uses Fortran (column-major) ordering for vectorization.

        For matrix multiplication C = X @ Y:
        - grad_X = kron(Y, I_m) where m = X.shape[0]
        - grad_Y = kron(I_n, X).T where n = Y.shape[1] (or 1 for vectors)

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if self.args[0].is_constant() or self.args[1].is_constant():
            return super(MulExpression, self)._grad(values)

        X = np.atleast_2d(values[0])
        Y = np.atleast_2d(values[1])

        # Handle 1D shapes: promote to 2D for consistent Kronecker computation
        x_shape = self.args[0].shape
        y_shape = self.args[1].shape

        # dot product of two vectors with shape (n,)
        if len(x_shape) == 1 and len(y_shape) == 1:
            # For 1D @ 1D -> scalar: grad is simply the other vector
            DX = sp.csc_array(values[1].reshape(-1, 1))
            DY = sp.csc_array(values[0].reshape(-1, 1))
            return [DX, DY]

        # For matrix @ vector, Y is (k,) -> treat as (k, 1)
        # Note: atleast_2d converts (k,) to (1, k), so we transpose to get (k, 1)
        if len(y_shape) == 1:
            Y = Y.T  # (1, k) from atleast_2d -> (k, 1)

        # For vector @ matrix, X is (k,) -> treat as (1, k)
        if len(x_shape) == 1:
            X = X  # already (1, k) from atleast_2d

        m = X.shape[0]  # rows of X
        n = Y.shape[1]  # cols of Y

        # grad_X = kron(Y, I_m) with shape (m*k, m*n)
        DX = sp.kron(Y, sp.eye_array(m), format='csc')

        # grad_Y = kron(I_n, X).T with shape (k*n, m*n)
        DY = sp.kron(sp.eye_array(n), X, format='csc').T

        return [DX, DY]

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Multiply the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        # Promote shapes for compatibility with CVXCanon
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if self.args[0].is_constant():
            return (lu.mul_expr(lhs, rhs, shape), [])
        elif self.args[1].is_constant():
            return (lu.rmul_expr(lhs, rhs, shape), [])
        else:
            raise DCPError("Product of two non-constant expressions is not "
                           "DCP.")


class multiply(MulExpression):
    """Multiplies two expressions elementwise."""

    OP_NAME = "*"

    def __init__(self, lh_expr, rh_expr) -> None:
        lh_expr, rh_expr = self.broadcast(lh_expr, rh_expr)
        super(multiply, self).__init__(lh_expr, rh_expr)

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?"""
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?"""
        return True

    def is_atom_quasiconvex(self) -> bool:
        return (
            self.args[0].is_constant() or self.args[1].is_constant()) or (
            self.args[0].is_nonneg() and self.args[1].is_nonpos()) or (
            self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_atom_quasiconcave(self) -> bool:
        return (
            self.args[0].is_constant() or self.args[1].is_constant()) or all(
            arg.is_nonneg() for arg in self.args) or all(
            arg.is_nonpos() for arg in self.args)

    def numeric(self, values):
        """Multiplies the values elementwise."""
        if sp.issparse(values[0]):
            return values[0].multiply(values[1])
        elif sp.issparse(values[1]):
            return values[1].multiply(values[0])
        else:
            return np.multiply(values[0], values[1])

    def validate_arguments(self):
        """Validate that the arguments are broadcastable."""
        np.broadcast(
            np.empty(self.args[0].shape, dtype=np.dtype([])),
            np.empty(self.args[1].shape, dtype=np.dtype([]))
        )

    def shape_from_args(self) -> Tuple[int, ...]:
        """Call np.broadcast on multiply arguments."""
        return np.broadcast_shapes(self.args[0].shape, self.args[1].shape)

    def is_psd(self) -> bool:
        """Is the expression a positive semidefinite matrix?
        """
        return (self.args[0].is_psd() and self.args[1].is_psd()) or \
               (self.args[0].is_nsd() and self.args[1].is_nsd())

    def is_nsd(self) -> bool:
        """Is the expression a negative semidefinite matrix?
        """
        return (self.args[0].is_psd() and self.args[1].is_nsd()) or \
               (self.args[0].is_nsd() and self.args[1].is_psd())

    def _grad(self, values):
        """Gives the (sub/super)gradient of elementwise multiply.

        For z = multiply(x, y), we have z[i] = x[i] * y[i].
        Gradient is diagonal: grad_x = diag(y), grad_y = diag(x).
        CVXPY convention: grad[i, j] = d(output[j]) / d(input[i])

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if self.args[0].is_constant() or self.args[1].is_constant():
            return super(multiply, self)._grad(values)

        X = values[0]
        Y = values[1]

        # Flatten in F-order for CVXPY convention
        x_flat = np.asarray(X).flatten(order='F')
        y_flat = np.asarray(Y).flatten(order='F')

        # Gradient is diagonal: grad_x[i, i] = y[i], grad_y[i, i] = x[i]
        DX = sp.diags(y_flat, format='csc')
        DY = sp.diags(x_flat, format='csc')

        return [DX, DY]

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Multiply the expressions elementwise.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of exprraints)
        """
        # promote if necessary.
        lhs = arg_objs[0]
        rhs = arg_objs[1]
        if self.args[0].is_constant():
            return (lu.multiply(lhs, rhs), [])
        elif self.args[1].is_constant():
            return (lu.multiply(rhs, lhs), [])
        else:
            raise DCPError("Product of two non-constant expressions is not "
                           "DCP.")


class DivExpression(BinaryOperator):
    """Division by scalar.

    Can be created by using the / operator of expression.
    """

    OP_NAME = "/"
    OP_FUNC = np.divide

    def __init__(self, lh_expr, rh_expr) -> None:
        lh_expr, rh_expr = self.broadcast(lh_expr, rh_expr)
        super(DivExpression, self).__init__(lh_expr, rh_expr)

    def numeric(self, values):
        """Divides numerator by denominator.
        """
        for i in range(2):
            if sp.issparse(values[i]):
                values[i] = values[i].toarray()
        return np.divide(values[0], values[1])

    def is_quadratic(self) -> bool:
        return self.args[0].is_quadratic() and self.args[1].is_constant()

    def has_quadratic_term(self) -> bool:
        """Can be a quadratic term if divisor is constant."""
        return self.args[0].has_quadratic_term() and self.args[1].is_constant()

    def is_qpwa(self) -> bool:
        return self.args[0].is_qpwa() and self.args[1].is_constant()

    def shape_from_args(self) -> Tuple[int, ...]:
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def is_atom_convex(self) -> bool:
        """Division is convex (affine) in its arguments only if
           the denominator is constant.
        """
        return self.args[1].is_constant()

    def is_atom_concave(self) -> bool:
        return self.is_atom_convex()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return True

    def is_atom_quasiconvex(self) -> bool:
        return self.args[1].is_nonneg() or self.args[1].is_nonpos()

    def is_atom_quasiconcave(self) -> bool:
        return self.is_atom_quasiconvex()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        if idx == 0:
            return self.args[1].is_nonneg()
        else:
            return self.args[0].is_nonpos()

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        if idx == 0:
            return self.args[1].is_nonpos()
        else:
            return self.args[0].is_nonneg()

    def graph_implementation(
        self, arg_objs, shape: Tuple[int, ...], data=None
    ) -> Tuple[lo.LinOp, List[Constraint]]:
        """Multiply the linear expressions.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return (lu.div_expr(arg_objs[0], arg_objs[1]), [])


def vdot(x, y):
    """
    Return the standard inner product (or "scalar product") of (x,y).

    Parameters
    ----------
    x : Expression, int, float, NumPy ndarray, or nested list thereof.
        The conjugate-linear argument to the inner product.
    y : Expression, int, float, NumPy ndarray, or nested list thereof.
        The linear argument to the inner product.

    Returns
    -------
    expr : Expression
        The standard inner product of (x,y), conjugate-linear in x.
        We always have ``expr.shape == ()``.

    Notes
    -----
    The arguments ``x`` and ``y`` can be nested lists; these lists
    will be flattened independently of one another.

    For example, if ``x = [[a],[b]]`` and  ``y = [c, d]`` (with ``a,b,c,d``
    real scalars), then this function returns an Expression representing
    ``a * c + b * d``.
    """
    x = deep_flatten(x)
    y = deep_flatten(y)
    prod = multiply(conj(x), y)
    return cvxpy_sum(prod)


def scalar_product(x, y):
    """
    Alias for vdot.
    """
    return vdot(x, y)


def outer(x, y):
    """
    Return the outer product of (x,y).

    Parameters
    ----------
    x : Expression, int, float, NumPy ndarray, or nested list thereof.
        Input is flattened if not already a vector.
        The linear argument to the outer product.
    y : Expression, int, float, NumPy ndarray, or nested list thereof.
        Input is flattened if not already a vector.
        The transposed-linear argument to the outer product.

    Returns
    -------
    expr : Expression
        The outer product of (x,y), linear in x and transposed-linear in y.
    """
    x = Expression.cast_to_const(x)
    if x.ndim > 1:
        raise ValueError("x must be a 1-d array.")
    y = Expression.cast_to_const(y)
    if y.ndim > 1:
        raise ValueError("y must be a 1-d array.")
    
    x = reshape(x, (x.size, 1), order='F')
    y = reshape(y, (1, y.size), order='F')
    return x @ y
