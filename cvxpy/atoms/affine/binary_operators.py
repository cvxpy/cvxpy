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


import sys

import cvxpy.interface as intf
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.promote import promote
from cvxpy.error import DCPError
import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
import numpy as np
import operator as op
import scipy.sparse as sp
if sys.version_info >= (3, 0):
    from functools import reduce


class BinaryOperator(AffAtom):
    """
    Base class for expressions involving binary operators. (other than addition)

    """

    OP_NAME = 'BINARY_OP'

    def __init__(self, lh_exp, rh_exp):
        super(BinaryOperator, self).__init__(lh_exp, rh_exp)

    def name(self):
        pretty_args = []
        for a in self.args:
            if isinstance(a, (AddExpression, DivExpression)):
                pretty_args.append('(' + a.name() + ')')
            else:
                pretty_args.append(a.name())
        return pretty_args[0] + ' ' + self.OP_NAME + ' ' + pretty_args[1]

    def numeric(self, values):
        """Applies the binary operator to the values.
        """
        return reduce(self.OP_FUNC, values)

    def sign_from_args(self):
        """Default to rules for times.
        """
        return u.sign.mul_sign(self.args[0], self.args[1])

    def is_imag(self):
        """Is the expression imaginary?
        """
        return (self.args[0].is_imag() and self.args[1].is_real()) or \
            (self.args[0].is_real() and self.args[1].is_imag())

    def is_complex(self):
        """Is the expression complex valued?
        """
        return (self.args[0].is_complex() or self.args[1].is_complex()) and \
            not (self.args[0].is_imag() and self.args[1].is_imag())


def matmul(lh_exp, rh_exp):
    """Matrix multipliction."""
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

    OP_NAME = "*"
    OP_FUNC = op.mul

    def numeric(self, values):
        """Matrix multiplication.
        """
        if self.args[0].shape == () or self.args[1].shape == () or \
           intf.is_sparse(values[0]) or intf.is_sparse(values[1]):
            return values[0] * values[1]
        else:
            return np.matmul(values[0], values[1])

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return u.shape.mul_shapes(self.args[0].shape, self.args[1].shape)

    def is_atom_convex(self):
        """Multiplication is convex (affine) in its arguments only if one of
           the arguments is constant.
        """
        return self.args[0].is_constant() or self.args[1].is_constant()

    def is_atom_concave(self):
        """If the multiplication atom is convex, then it is affine.
        """
        return self.is_atom_convex()

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return False

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.args[1-idx].is_nonneg()

    def is_decr(self, idx):
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

        # DX = [diag(Y11), diag(Y12), ...]
        #      [diag(Y21), diag(Y22), ...]
        #      [   ...        ...     ...]
        DX = sp.dok_matrix((DX_rows, cols))
        for k in range(self.args[0].shape[0]):
            DX[k::self.args[0].shape[0], k::self.args[0].shape[0]] = Y
        DX = sp.csc_matrix(DX)
        cols = 1 if len(self.args[1].shape) == 1 else self.args[1].shape[1]
        DY = sp.block_diag([X.T for k in range(cols)], 'csc')

        return [DX, DY]

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
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
        if lu.is_const(lhs):
            return (lu.mul_expr(lhs, rhs, shape), [])
        elif lu.is_const(rhs):
            return (lu.rmul_expr(lhs, rhs, shape), [])
        else:
            raise DCPError("Product of two non-constant expressions is not "
                           "DCP.")


class multiply(MulExpression):
    """ Multiplies two expressions elementwise.
    """

    def __init__(self, lh_expr, rh_expr):
        lh_expr = multiply.cast_to_const(lh_expr)
        rh_expr = multiply.cast_to_const(rh_expr)
        if lh_expr.is_scalar() and not rh_expr.is_scalar():
            lh_expr = promote(lh_expr, rh_expr.shape)
        elif rh_expr.is_scalar() and not lh_expr.is_scalar():
            rh_expr = promote(rh_expr, lh_expr.shape)
        super(multiply, self).__init__(lh_expr, rh_expr)

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return True

    def is_atom_quasiconvex(self):
        return (
            self.args[0].is_constant() or self.args[1].is_constant()) or (
            self.args[0].is_nonneg() and self.args[1].is_nonpos()) or (
            self.args[0].is_nonpos() and self.args[1].is_nonneg())

    def is_atom_quasiconcave(self):
        return (
            self.args[0].is_constant() or self.args[1].is_constant()) or all(
            arg.is_nonneg() for arg in self.args) or all(
            arg.is_nonpos() for arg in self.args)

    def numeric(self, values):
        """Multiplies the values elementwise.
        """
        if sp.issparse(values[0]):
            return values[0].multiply(values[1])
        elif sp.issparse(values[1]):
            return values[1].multiply(values[0])
        else:
            return np.multiply(values[0], values[1])

    def shape_from_args(self):
        """The sum of the argument dimensions - 1.
        """
        return u.shape.sum_shapes([arg.shape for arg in self.args])

    def is_psd(self):
        """Is the expression a positive semidefinite matrix?
        """
        return (self.args[0].is_psd() and self.args[1].is_psd()) or \
               (self.args[0].is_nsd() and self.args[1].is_nsd())

    def is_nsd(self):
        """Is the expression a negative semidefinite matrix?
        """
        return (self.args[0].is_psd() and self.args[1].is_nsd()) or \
               (self.args[0].is_nsd() and self.args[1].is_psd())

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
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
        if lu.is_const(lhs):
            return (lu.multiply(lhs, rhs), [])
        elif lu.is_const(rhs):
            return (lu.multiply(rhs, lhs), [])
        else:
            raise DCPError("Product of two non-constant expressions is not "
                           "DCP.")


class DivExpression(multiply):
    """Division by scalar.

    Can be created by using the / operator of expression.
    """

    OP_NAME = "/"
    OP_FUNC = np.divide

    def numeric(self, values):
        """Divides numerator by denominator.
        """
        for i in range(2):
            if sp.issparse(values[i]):
                values[i] = values[i].todense().A
        return np.divide(values[0], values[1])

    def is_quadratic(self):
        return self.args[0].is_quadratic() and self.args[1].is_constant()

    def is_qpwa(self):
        return self.args[0].is_qpwa() and self.args[1].is_constant()

    def shape_from_args(self):
        """Returns the (row, col) shape of the expression.
        """
        return self.args[0].shape

    def is_atom_convex(self):
        """Division is convex (affine) in its arguments only if
           the denominator is constant.
        """
        return self.args[1].is_constant()

    def is_atom_concave(self):
        return self.is_atom_convex()

    def is_atom_log_log_convex(self):
        """Is the atom log-log convex?
        """
        return True

    def is_atom_log_log_concave(self):
        """Is the atom log-log concave?
        """
        return True

    def is_atom_quasiconvex(self):
        return self.args[1].is_nonneg() or self.args[1].is_nonpos()

    def is_atom_quasiconcave(self):
        return self.is_atom_quasiconvex()

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        if idx == 0:
            return self.args[1].is_nonneg()
        else:
            return self.args[0].is_nonpos()

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        if idx == 0:
            return self.args[1].is_nonpos()
        else:
            return self.args[0].is_nonneg()

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
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
