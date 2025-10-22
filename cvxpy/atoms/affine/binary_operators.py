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
from cvxpy.atoms.affine.conj import conj
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.reshape import deep_flatten, reshape
from cvxpy.atoms.affine.sum import sum as cvxpy_sum
from cvxpy.constraints.constraint import Constraint
from cvxpy.error import DCPError
from cvxpy.expressions.constants.parameter import (
    is_param_affine,
    is_param_free,
)
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


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

    def numeric(self, values):
        """Matrix multiplication.
        """
        if values[0].shape == () or values[1].shape == ():
            return values[0] * values[1]
        else:
            return values[0] @ values[1]

    def validate_arguments(self):
        """Validate that the arguments can be multiplied together."""
        if self.args[0].ndim > 2 or self.args[1].ndim > 2:
            raise ValueError("Multiplication with N-d arrays is not yet supported")
    
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

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        if self.args[0].is_constant() or self.args[1].is_constant():
            return super(MulExpression, self)._grad(values)

        # TODO(akshayka): Verify that the following code is correct for
        # non-affine arguments.
        X = values[0]
        Y = values[1]

        DX_rows = self.args[0].size
        cols = self.args[0].size

        # dot product of two vectors with shape (n,)
        if len(self.args[0].shape) == 1 and len(self.args[1].shape) == 1:
            DX = sp.csc_array(Y.reshape(-1, 1))  # y as column vector
            DY = sp.csc_array(X.reshape(-1, 1))  # x as column vector
            return [DX, DY]

        # DX = [diag(Y11), diag(Y12), ...]
        #      [diag(Y21), diag(Y22), ...]
        #      [   ...        ...     ...]
        DX = sp.dok_array((DX_rows, cols))
        for k in range(self.args[0].shape[0]):
            DX[k::self.args[0].shape[0], k::self.args[0].shape[0]] = Y
        DX = sp.csc_array(DX)
        cols = 1 if len(self.args[1].shape) == 1 else self.args[1].shape[1]
        DY = sp.block_diag([np.atleast_2d(X.T) for k in range(cols)], "csc")

        return [DX, DY]
    
    def _verify_hess_vec_args(self):
        x = self.args[0]
        y = self.args[1]
        if x.size != y.size:
            return False

        if x.is_constant() and y.is_constant():
            return False

        # one of the following must be true:
        # 1. both arguments are variables
        # 2. one argument is a constant
        # 3. one argument is a Promote of a variable and the other is a variable
        both_are_variables = isinstance(x, Variable) and isinstance(y, Variable)
        one_is_constant = x.is_constant() or y.is_constant()
        x_is_promote = type(x) == Promote and isinstance(y, Variable)
        y_is_promote = type(y) == Promote and isinstance(x, Variable)

        if not (both_are_variables or one_is_constant or x_is_promote or y_is_promote):
            return False
        
        if both_are_variables and x.id == y.id:
            return False

        return True

    def _hess_vec(self, vec):
        x = self.args[0]
        y = self.args[1]
        
        # constant * atom
        if x.is_constant(): 
            y_hess_vec = y.hess_vec(x.value.flatten(order='F') * vec)
            return y_hess_vec
        
        # atom * constant
        if y.is_constant():
            x_hess_vec = x.hess_vec(y.value.flatten(order='F') * vec)
            return x_hess_vec

        # x * y with x a scalar variable, y a vector variable
        if not isinstance(x, Variable) and x.is_affine():
            assert(type(x) == Promote)
            x_var = x.args[0] # here x is a Promote because of how we canonicalize
            return {(x_var, y): vec, (y, x_var): vec}
        
        # x * y with x a vector variable, y a scalar
        if not isinstance(y, Variable) and y.is_affine():
            assert(type(y) == Promote)
            y_var = y.args[0] # here y is a Promote because of how we canonicalize
            return {(x, y_var): vec, (y_var, x): vec}
        
        # if we arrive here both arguments are variables of the same size
        return {(x, y): np.diag(vec), (y, x): np.diag(vec)}

    # todo: always assume it is A @ x for now where A is constant and x is variable
    def _jacobian(self):
        A = self.args[0].value
        x = self.args[1]

        if not isinstance(A, sp.coo_matrix):
            A = sp.coo_matrix(A)
        
        return {x: (A.row, A.col, A.data)}

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
        """Compute the gradient of elementwise multiplication w.r.t. each argument.
    
        For z = x * y (elementwise), returns:
        - dz/dx = diag(y)
        - dz/dy = diag(x)
    
        Args:
            values: A list of numeric values for the arguments [x, y].
    
        Returns:
            A list of SciPy CSC sparse matrices [DX, DY].
        """
        x = values[0]
        y = values[1]
        # Flatten in case inputs are not 1D
        x = np.asarray(x).flatten(order='F')
        y = np.asarray(y).flatten(order='F')
        DX = sp.diags(y, format='csc')
        DY = sp.diags(x, format='csc')
        return [DX, DY]
    
    def _verify_hess_vec_args(self):
        x = self.args[0]
        y = self.args[1]
        if x.size != y.size:
            return False

        if x.is_constant() and y.is_constant():
            return False

        # one of the following must be true:
        # 1. both arguments are variables
        # 2. one argument is a constant
        # 3. one argument is a Promote of a variable and the other is a variable
        both_are_variables = isinstance(x, Variable) and isinstance(y, Variable)
        one_is_constant = x.is_constant() or y.is_constant()
        x_is_promote = type(x) == Promote and isinstance(y, Variable)
        y_is_promote = type(y) == Promote and isinstance(x, Variable)

        if not (both_are_variables or one_is_constant or x_is_promote or y_is_promote):
            return False
        
        if both_are_variables and x.id == y.id:
            return False

        return True

    def _hess_vec(self, vec):
        x = self.args[0]
        y = self.args[1]
        
        # constant * atom
        if x.is_constant(): 
            y_hess_vec = y.hess_vec(x.value * vec)
            return y_hess_vec
        
        # atom * constant
        if y.is_constant():
            x_hess_vec = x.hess_vec(y.value * vec)
            return x_hess_vec

        # x * y with x a scalar variable, y a vector variable
        if not isinstance(x, Variable) and x.is_affine():
            assert(type(x) == Promote)
            x_var = x.args[0] # here x is a Promote because of how we canonicalize
            zeros_x = np.zeros(x_var.size, dtype=int)
            cols = np.arange(y.size)
            return {(x_var, y): (zeros_x, cols, vec),
                    (y, x_var): (cols, zeros_x, vec)}
        
        # x * y with x a vector variable, y a scalar
        if not isinstance(y, Variable) and y.is_affine():
            assert(type(y) == Promote)
            y_var = y.args[0] # here y is a Promote because of how we canonicalize
            zeros_y = np.zeros(y_var.size, dtype=int)
            cols = np.arange(x.size)
            return {(x, y_var): (cols, zeros_y, vec),
                    (y_var, x): (zeros_y, cols, vec)}
        
        # if we arrive here both arguments are variables of the same size
        rows = np.arange(x.size)
        cols = np.arange(x.size)
        return {(x, y): (rows, cols, vec), (y, x): (rows, cols, vec)}

    def _verify_jacobian_args(self):
        return self._verify_hess_vec_args()

    def _jacobian(self):
        x = self.args[0]
        y = self.args[1]

        if x.is_constant():
            dy = y.jacobian()
            for k in dy:
                rows, cols, vals = dy[k]
                # this is equivalent to forming the matrix defined
                # rows, cols, vals and scaling each row i by y.value[i]
                dy[k] = (rows, cols, np.atleast_1d(x.value)[rows] * vals)
            return dy

        if y.is_constant():
            dx = x.jacobian()
            for k in dx:
                rows, cols, vals = dx[k]
                dx[k] = (rows, cols, np.atleast_1d(y.value)[rows] * vals)
            return dx

        if not isinstance(x, Variable) and x.is_affine():
            assert(type(x) == Promote)
            x_var = x.args[0] # here x is a Promote because of how we canonicalize
            idxs = np.arange(y.size)
            return {(x_var): (idxs, np.zeros(y.size, dtype=int), y.value),
                    (y): (idxs, idxs, x.value)}
        
        # x * y with x a vector variable, y a scalar
        if not isinstance(y, Variable) and y.is_affine():
            assert(type(y) == Promote)
            y_var = y.args[0] # here y is a Promote because of how we canonicalize
            idxs = np.arange(x.size)
            return {(x): (idxs, idxs, y.value),
                    (y_var): (idxs, np.zeros(x.size, dtype=int), x.value)}
        
        # here both are variables
        idxs = np.arange(x.size)
        jacobian_dict = {x: (idxs, idxs, y.value.flatten(order='F')),
                        y: (idxs, idxs, x.value.flatten(order='F'))}
        return jacobian_dict

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
    
    def point_in_domain(self):
        return np.ones(self.args[1].shape)

    def _verify_hess_vec_args(self):
        raise RuntimeError("The _verify_hess_vec_args method of"
                           " the division atom should never be called.")

    def _hess_vec(self, vec):
        raise RuntimeError("The hess_vec method of the division atom should never "
                           "be called.")

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
