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

from cvxpy.lin_ops.lin_to_matrix import get_matrix
from cvxpy.lin_ops.lin_utils import *
from cvxpy.lin_ops.lin_op import *
from cvxpy.expressions.constants import Parameter
import numpy as np
import scipy.sparse as sp
import unittest
from base_test import BaseTest

class test_lin_ops(BaseTest):
    """ Unit tests for the lin_ops module. """

    def test_variables(self):
        """Test creating a variable.
        """
        var = create_var((5, 4))
        self.assertEqual(var.var_size, (5, 4))
        self.assertEqual(var.scalar_coeff, 1.0)
        self.assertEqual(var.type, EYE_MUL)

    # def test_param(self):
    #     """Test creating a parameter.
    #     """
    #     A = Parameter(5, 4)
    #     var = create_param(A, (5, 4))
    #     self.assertEqual(var.var_size, (5, 4))
    #     self.assertEqual(var.scalar_coeff, 1.0)
    #     self.assertEqual(var.type, PARAM)

    def test_constant(self):
        """Test creating a constant.
        """
        # Scalar constant.
        size = (1, 1)
        mat = create_const(1.0, size)
        self.assertEqual(mat.var_size, size)
        self.assertEqual(mat.scalar_coeff, 1.0)
        self.assertEqual(mat.type, SCALAR_CONST)
        assert mat.data == 1.0

        # Dense matrix constant.
        size = (5, 4)
        mat = create_const(np.ones(size), size)
        self.assertEqual(mat.var_size, size)
        self.assertEqual(mat.scalar_coeff, 1.0)
        self.assertEqual(mat.type, DENSE_CONST)
        assert (mat.data == np.ones(size)).all()

        # Sparse matrix constant.
        size = (5, 5)
        mat = create_const(sp.eye(5), size)
        self.assertEqual(mat.var_size, size)
        self.assertEqual(mat.scalar_coeff, 1.0)
        self.assertEqual(mat.type, SPARSE_CONST)
        assert (mat.data.todense() == sp.eye(5).todense()).all()

    def test_add_expr(self):
        """Test adding lin expr.
        """
        size = (5, 4)
        x = create_var(size)
        y = create_var(size)
        x_expr = LinExpr([x], size)
        y_expr = LinExpr([y], size)
        # Expanding dict.
        add_expr = sum_expr([x_expr, y_expr])
        terms = add_expr.terms
        assert len(terms) == 2
        self.assertItemsEqual([x, y], terms)

    def test_neg_term(self):
        """Test negating a term.
        """
        var = create_var((5, 4))
        term = neg_term(var)
        self.assertEqual(term.var_size, (5, 4))
        self.assertEqual(term.scalar_coeff, -1.0)
        self.assertEqual(term.type, EYE_MUL)

    def test_mul_term(self):
        """Test multiplying terms by a constant.
        """
        size = (5, 5)
        # Eye times constants.
        x = create_var(size)
        # Scalar
        A = create_const(2.0, (1, 1))
        term = mul_term(A, x)
        self.assertEqual(term.var_size, size)
        self.assertEqual(term.scalar_coeff, 2.0)
        self.assertEqual(term.type, EYE_MUL)
        # Dense
        A = create_const(np.ones(size), size)
        term = mul_term(A, x)
        self.assertEqual(term.scalar_coeff, 1.0)
        self.assertEqual(term.type, DENSE_MUL)
        self.assertItemsAlmostEqual(term.data, np.ones(size))
        # Sparse
        A = create_const(sp.eye(5), size)
        term = mul_term(A, x)
        self.assertEqual(term.scalar_coeff, 1.0)
        self.assertEqual(term.type, SPARSE_MUL)
        self.assertItemsAlmostEqual(term.data.todense(),
            sp.eye(5).todense())
        # Parameter
        param = Parameter(*size)
        param.value = np.ones(size)
        A = LinOp(PARAM, CONSTANT_ID, size, 1.0, param)
        term = mul_term(A, x)
        self.assertEqual(term.scalar_coeff, 1.0)
        self.assertEqual(term.type, PARAM_MUL)
        self.assertItemsAlmostEqual(term.data.value, param.value)

        # Multiplying by a scalar.
        op = LinOp(TRANSPOSE, x.var_id, x.var_size, 1.0, None)
        A = create_const(5.0, (1, 1))
        term = mul_term(A, op)
        self.assertEqual(term.scalar_coeff, 5.0)
        self.assertEqual(term.type, TRANSPOSE)

        # Dense * Dense
        value = np.ones(size)
        A = create_const(value, size)
        term = mul_term(A, x)
        term = mul_term(A, term)
        self.assertEqual(term.scalar_coeff, 1.0)
        self.assertEqual(term.type, DENSE_MUL)
        self.assertItemsAlmostEqual(term.data, value*value)

        term = mul_term(A, A)
        self.assertEqual(term.type, DENSE_CONST)
        self.assertItemsAlmostEqual(term.data, value*value)

        # Sparse * Sparse
        value = 3*sp.eye(5)
        A = create_const(value, size)
        term = mul_term(A, x)
        term = mul_term(A, term)
        self.assertEqual(term.scalar_coeff, 1.0)
        self.assertEqual(term.type, SPARSE_MUL)
        self.assertItemsAlmostEqual(term.data.todense(),
            (value*value).todense())

        term = mul_term(A, A)
        self.assertEqual(term.type, SPARSE_CONST)
        self.assertItemsAlmostEqual(term.data.todense(),
            (value*value).todense())

        # Dense * Sparse
        value = 3*sp.eye(5)
        A = create_const(value, size)
        B = create_const(np.ones(size), size)
        term = mul_term(A, x)
        term = mul_term(B, term)
        self.assertEqual(term.type, DENSE_MUL)
        self.assertItemsAlmostEqual(term.data, np.ones(size)*value)

    def test_mul_expr(self):
        """Test multiplying an expression by a constant.
        """
        size = (5, 5)
        x = create_var(size)
        y = create_var(size)
        # No constraints needed.
        expr = LinExpr([x, y], size)
        value = np.ones(size)
        A = create_const(value, size)
        product, constr = mul_expr(A, expr, size)
        assert len(constr) == 0
        assert len(product.terms) == 2
        self.assertEqual(product.terms[0].type, DENSE_MUL)

        # Constraint needed.
        op = LinOp(TRANSPOSE, x.var_id, x.var_size, 1.0, None)
        expr = LinExpr([x, y, op], size)
        A = create_const(value, size)
        product, constr = mul_expr(A, expr, size)
        assert len(constr) == 1
        assert len(constr[0].expr.terms) == 4
        assert len(product.terms) == 1
        self.assertEqual(product.terms[0].type, DENSE_MUL)

    def test_eq_constr(self):
        """Test creating an equality constraint.
        """
        size = (5, 5)
        x = create_var(size)
        y = create_var(size)
        lh_expr = LinExpr([x, y], size)
        value = np.ones(size)
        A = create_const(value, size)
        rh_expr = LinExpr([A], size)
        constr = create_eq(lh_expr, rh_expr)
        self.assertEqual(constr.size, size)
        assert len(constr.expr.terms) == 3
        coeffs = [t.scalar_coeff for t in constr.expr.terms]
        self.assertItemsEqual([1, 1, -1], coeffs)

    def test_leq_constr(self):
        """Test creating a less than or equal constraint.
        """
        size = (5, 5)
        x = create_var(size)
        y = create_var(size)
        lh_expr = LinExpr([x, y], size)
        value = np.ones(size)
        A = create_const(value, size)
        rh_expr = LinExpr([A], size)
        constr = create_leq(lh_expr, rh_expr)
        self.assertEqual(constr.size, size)
        assert len(constr.expr.terms) == 3
        coeffs = [t.scalar_coeff for t in constr.expr.terms]
        self.assertItemsEqual([1, 1, -1], coeffs)

    def test_get_matrix(self):
        """Test the get_matrix function.
        """
        size = (5, 4)
        # Eye
        x = create_var(size)
        mat = get_matrix(x)
        self.assertItemsAlmostEqual(mat.todense(), sp.eye(20).todense())
        # Eye with scalar mult.
        x = create_var(size)
        A = create_const(5, (1, 1))
        mat = get_matrix(mul_term(A, x))
        self.assertItemsAlmostEqual(mat.todense(), 5*sp.eye(20).todense())
        # Promoted
        x = create_var((1, 1))
        A = create_const(np.ones(size), size)
        mat = get_matrix(mul_term(A, x))
        self.assertEqual(mat.shape, (20, 1))
        self.assertItemsAlmostEqual(mat, np.ones((20, 1)))
        # Normal
        size = (5, 5)
        x = create_var((5, 1))
        A = create_const(np.ones(size), size)
        mat = get_matrix(mul_term(A, x))
        self.assertEqual(mat.shape, (5, 5))
        self.assertItemsAlmostEqual(mat.todense(), A.data)
        # Blocks
        size = (5, 5)
        x = create_var(size)
        A = create_const(np.ones(size), size)
        mat = get_matrix(mul_term(A, x))
        self.assertEqual(mat.shape, (25, 25))
        self.assertItemsAlmostEqual(mat.todense(),
         sp.block_diag(5*[np.ones(size)]).todense())
        # Scalar constant
        size = (1, 1)
        A = create_const(5, size)
        mat = get_matrix(A)
        self.assertEqual(mat.shape, (1, 1))
        self.assertItemsAlmostEqual(mat, np.array(5))
        # Dense constant
        size = (5, 4)
        A = create_const(np.ones(size), size)
        mat = get_matrix(A)
        self.assertEqual(mat.shape, (20, 1))
        self.assertItemsAlmostEqual(mat, np.ones((20, 1)))
        # Sparse constant
        size = (5, 5)
        A = create_const(sp.eye(5), size)
        mat = get_matrix(A)
        self.assertEqual(mat.shape, (25, 1))
        test_mat = np.zeros((25, 1))
        for i in range(5):
            test_mat[i*5 + i] = 1
        self.assertItemsAlmostEqual(mat, test_mat)
        # Parameter
        size = (5, 4)
        param = Parameter(*size)
        param.value = np.ones(size)
        A = LinOp(PARAM, CONSTANT_ID, size, 2.0, param)
        mat = get_matrix(A)
        self.assertEqual(mat.shape, (20, 1))
        self.assertItemsAlmostEqual(mat, 2*np.ones((20, 1)))

    # def test_add_terms(self):
    #     """Test adding lin ops. Assume ids match.
    #     """
    #     size = (5, 4)
    #     # Add variables.
    #     x = create_var(size)
    #     op = add_terms(x, x)
    #     self.assertEqual(op.var_id, x.var_id)
    #     self.assertEqual(op.var_size, size)
    #     self.assertEqual(op.scalar_coeff, 2.0)
    #     self.assertEqual(op.type, EYE_MUL)

    #     # Add dense * var.
    #     op = LinOp(DENSE_MUL, x.var_id, x.var_size, 2.0, np.ones((5, 5)))
    #     add_op = add_terms(op, op)
    #     self.assertEqual(add_op.var_id, x.var_id)
    #     self.assertEqual(add_op.var_size, size)
    #     self.assertEqual(add_op.scalar_coeff, 1.0)
    #     self.assertEqual(add_op.type, DENSE_MUL)
    #     self.assertItemsAlmostEqual(add_op.data, 4*np.ones((5, 5)))

    #     # Add sparse * var.
    #     op = LinOp(SPARSE_MUL, x.var_id, x.var_size, 2.0, sp.eye(5))
    #     add_op = add_terms(op, op)
    #     self.assertEqual(add_op.var_id, x.var_id)
    #     self.assertEqual(add_op.var_size, size)
    #     self.assertEqual(add_op.scalar_coeff, 1.0)
    #     self.assertEqual(add_op.type, SPARSE_MUL)
    #     self.assertItemsAlmostEqual(add_op.data.todense(), 4*sp.eye(5).todense())

    #     # Add transpose.
    #     op = LinOp(TRANSPOSE, x.var_id, x.var_size, 1.0, None)
    #     add_op = add_terms(op, op)
    #     self.assertEqual(add_op.var_id, x.var_id)
    #     self.assertEqual(add_op.var_size, size)
    #     self.assertEqual(add_op.scalar_coeff, 2.0)
    #     self.assertEqual(add_op.type, TRANSPOSE)

    #     # Add transpose to eye.
    #     size = (5, 5)
    #     x = create_var(size)
    #     op = LinOp(TRANSPOSE, x.var_id, x.var_size, 1.0, None)
    #     add_op = add_terms(op, x)
    #     assert add_op is None
