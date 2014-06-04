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

from cvxpy.lin_ops.lin_to_matrix import get_coefficients
from cvxpy.lin_ops.lin_utils import *
from cvxpy.lin_ops.lin_op import *
from cvxpy.expressions.constants import Parameter
import cvxpy.interface as intf
import numpy as np
import scipy.sparse as sp
import unittest
from base_test import BaseTest

class test_lin_ops(BaseTest):
    """ Unit tests for the lin_ops module. """

    def test_variables(self):
        """Test creating a variable.
        """
        var = create_var((5, 4), var_id=1)
        self.assertEqual(var.size, (5, 4))
        self.assertEqual(var.data, 1)
        self.assertEqual(len(var.args), 0)
        self.assertEqual(var.type, VARIABLE)

    def test_param(self):
        """Test creating a parameter.
        """
        A = Parameter(5, 4)
        var = create_param(A, (5, 4))
        self.assertEqual(var.size, (5, 4))
        self.assertEqual(len(var.args), 0)
        self.assertEqual(var.type, PARAM)

    def test_constant(self):
        """Test creating a constant.
        """
        # Scalar constant.
        size = (1, 1)
        mat = create_const(1.0, size)
        self.assertEqual(mat.size, size)
        self.assertEqual(len(mat.args), 0)
        self.assertEqual(mat.type, SCALAR_CONST)
        assert mat.data == 1.0

        # Dense matrix constant.
        size = (5, 4)
        mat = create_const(np.ones(size), size)
        self.assertEqual(mat.size, size)
        self.assertEqual(len(mat.args), 0)
        self.assertEqual(mat.type, DENSE_CONST)
        assert (mat.data == np.ones(size)).all()

        # Sparse matrix constant.
        size = (5, 5)
        mat = create_const(sp.eye(5), size, sparse=True)
        self.assertEqual(mat.size, size)
        self.assertEqual(len(mat.args), 0)
        self.assertEqual(mat.type, SPARSE_CONST)
        assert (mat.data.todense() == sp.eye(5).todense()).all()

    def test_add_expr(self):
        """Test adding lin expr.
        """
        size = (5, 4)
        x = create_var(size)
        y = create_var(size)
        # Expanding dict.
        add_expr = sum_expr([x, y])
        self.assertEqual(add_expr.size, size)
        assert len(add_expr.args) == 2

    def test_get_vars(self):
        """Test getting vars from an expression.
        """
        size = (5, 4)
        x = create_var(size)
        y = create_var(size)
        A = create_const(np.ones(size), size)
        # Expanding dict.
        add_expr = sum_expr([x, y, A])
        vars_ = get_expr_vars(add_expr)
        self.assertItemsEqual(vars_, [(x.data, size), (y.data, size)])

    def test_neg_expr(self):
        """Test negating an expression.
        """
        size = (5, 4)
        var = create_var(size)
        expr = neg_expr(var)
        assert len(expr.args) == 1
        self.assertEqual(expr.size, size)
        self.assertEqual(expr.type, NEG)

    def test_eq_constr(self):
        """Test creating an equality constraint.
        """
        size = (5, 5)
        x = create_var(size)
        y = create_var(size)
        lh_expr = sum_expr([x, y])
        value = np.ones(size)
        rh_expr = create_const(value, size)
        constr = create_eq(lh_expr, rh_expr)
        self.assertEqual(constr.size, size)
        vars_ = get_expr_vars(constr.expr)
        self.assertItemsEqual(vars_, [(x.data, size), (y.data, size)])

    def test_leq_constr(self):
        """Test creating a less than or equal constraint.
        """
        size = (5, 5)
        x = create_var(size)
        y = create_var(size)
        lh_expr = sum_expr([x, y])
        value = np.ones(size)
        rh_expr = create_const(value, size)
        constr = create_leq(lh_expr, rh_expr)
        self.assertEqual(constr.size, size)
        vars_ = get_expr_vars(constr.expr)
        self.assertItemsEqual(vars_, [(x.data, size), (y.data, size)])

    def test_get_coefficients(self):
        """Test the get_coefficients function.
        """
        size = (5, 4)
        # Eye
        x = create_var(size)
        coeffs = get_coefficients(x)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(id_, x.data)
        self.assertItemsAlmostEqual(mat.todense(), sp.eye(20).todense())
        # Eye with scalar mult.
        x = create_var(size)
        A = create_const(5, (1, 1))
        coeffs = get_coefficients(mul_expr(A, x, size))
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertItemsAlmostEqual(mat.todense(), 5*sp.eye(20).todense())
        # Promoted
        x = create_var((1, 1))
        coeffs = get_coefficients(promote(x, size))
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (20, 1))
        self.assertItemsAlmostEqual(mat, np.ones((20, 1)))
        # Normal
        size = (5, 5)
        x = create_var((5, 1))
        A = create_const(np.ones(size), size)
        coeffs = get_coefficients(mul_expr(A, x, (5, 1)))
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (5, 5))
        self.assertItemsAlmostEqual(mat.todense(), A.data)
        # Blocks
        size = (5, 5)
        x = create_var(size)
        A = create_const(np.ones(size), size)
        coeffs = get_coefficients(mul_expr(A, x, size))
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (25, 25))
        self.assertItemsAlmostEqual(mat.todense(),
         sp.block_diag(5*[np.ones(size)]).todense())
        # Scalar constant
        size = (1, 1)
        A = create_const(5, size)
        coeffs = get_coefficients(A)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(intf.size(mat), (1, 1))
        self.assertEqual(mat, 5)
        # Dense constant
        size = (5, 4)
        A = create_const(np.ones(size), size)
        coeffs = get_coefficients(A)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (size[0]*size[1], 1))
        self.assertItemsAlmostEqual(mat, np.ones(size))
        # Sparse constant
        size = (5, 5)
        A = create_const(sp.eye(5), size)
        coeffs = get_coefficients(A)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (size[0]*size[1], 1))
        self.assertItemsAlmostEqual(mat, sp.eye(5).todense())
        # Parameter
        size = (5, 4)
        param = Parameter(*size)
        param.value = np.ones(size)
        A = create_param(param, size)
        coeffs = get_coefficients(A)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (size[0]*size[1], 1))
        self.assertItemsAlmostEqual(mat, param.value)

    def test_transpose(self):
        """Test transpose op and coefficients.
        """
        size = (5, 4)
        x = create_var(size)
        expr = transpose(x)
        self.assertEqual(expr.size, (4, 5))
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        test_mat = np.mat(range(20)).T
        self.assertItemsAlmostEqual((mat*test_mat).reshape((4, 5), order='F'),
            test_mat.reshape(size, order='F').T)

    def test_index(self):
        """Test the get_coefficients function for index.
        """
        size = (5, 4)
        # Eye
        key = (slice(0,2,None), slice(0,2,None))
        x = create_var(size)
        expr = index(x, (2, 2), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(id_, x.data)
        self.assertEqual(mat.shape, (4, 20))
        test_mat = np.mat(range(20)).T
        self.assertItemsAlmostEqual((mat*test_mat).reshape((2, 2), order='F'),
            test_mat.reshape(size, order='F')[key])
        # Eye with scalar mult.
        key = (slice(0,2,None), slice(0,2,None))
        x = create_var(size)
        A = create_const(5, (1, 1))
        expr = mul_expr(A, x, size)
        expr = index(expr, (2, 2), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        test_mat = np.mat(range(20)).T
        self.assertItemsAlmostEqual((mat*test_mat).reshape((2, 2), order='F'),
            5*test_mat.reshape(size, order='F')[key])
        # Promoted
        key = (slice(0,2,None), slice(0,2,None))
        x = create_var((1, 1))
        value = np.array(range(20)).reshape(size)
        A = create_const(value, size)
        prom_x = promote(x, (size[1], 1))
        expr = mul_expr(A, diag_vec(prom_x), size)
        expr = index(expr, (2, 2), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (4, 1))
        self.assertItemsAlmostEqual(mat, value[key])
        # Normal
        size = (5, 5)
        key = (slice(0,2,None), slice(0,1,None))
        x = create_var((5, 1))
        A = create_const(np.ones(size), size)
        expr = mul_expr(A, x, (5, 1))
        expr = index(expr, (2, 1), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (2, 5))
        self.assertItemsAlmostEqual(mat.todense(), A.data[slice(0,2,None)])
        # Blocks
        size = (5, 5)
        key = (slice(0,2,None), slice(0,2,None))
        x = create_var(size)
        value = np.array(range(25)).reshape(size)
        A = create_const(value, size)
        expr = mul_expr(A, x, size)
        expr = index(expr, (2, 2), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (4, 25))
        test_mat = np.mat(range(25)).T
        self.assertItemsAlmostEqual((mat*test_mat).reshape((2, 2), order='F'),
            (A.data*test_mat.reshape(size, order='F'))[key])
        # Scalar constant
        size = (1, 1)
        A = create_const(5, size)
        key = (slice(0,1,None), slice(0,1,None))
        expr = index(A, (1, 1), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(intf.size(mat), (1, 1))
        self.assertEqual(mat, 5)
        # Dense constant
        size = (5, 4)
        key = (slice(0,2,None), slice(0,1,None))
        value = np.array(range(20)).reshape(size)
        A = create_const(value, size)
        expr = index(A, (2, 1), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (2, 1))
        self.assertItemsAlmostEqual(mat, value[key])
        # Sparse constant
        size = (5, 5)
        key = (slice(0,2,None), slice(0,1,None))
        A = create_const(sp.eye(5), size)
        expr = index(A, (2, 1), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (2, 1))
        self.assertItemsAlmostEqual(mat, sp.eye(5).todense()[key])
        # Parameter
        size = (5, 4)
        key = (slice(0,2,None), slice(0,1,None))
        param = Parameter(*size)
        value = np.array(range(20)).reshape(size)
        param.value = value
        A = create_param(param, size)
        expr = index(A, (2, 1), key)
        coeffs = get_coefficients(expr)
        assert len(coeffs) == 1
        id_, mat = coeffs[0]
        self.assertEqual(mat.shape, (2, 1))
        self.assertItemsAlmostEqual(mat, param.value[key])


    def test_sum_entries(self):
        """Test sum entries op.
        """
        size = (5, 5)
        x = create_var(size)
        expr = sum_entries(x)
        self.assertEqual(expr.size, (1, 1))
        self.assertEqual(len(expr.args), 1)
        self.assertEqual(expr.type, lo.SUM_ENTRIES)
