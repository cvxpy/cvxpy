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

from cvxpy.lin_ops.lin_utils import *
from cvxpy.lin_ops.lin_op import *
from cvxpy.expressions.constants import Parameter
import cvxpy.interface as intf
import numpy as np
import scipy.sparse as sp
import unittest
from cvxpy.tests.base_test import BaseTest
import sys
PY2 = sys.version_info < (3, 0)


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
        ref = [(x.data, size), (y.data, size)]
        if PY2:
            self.assertItemsEqual(vars_, ref)
        else:
            self.assertCountEqual(vars_, ref)

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
        ref = [(x.data, size), (y.data, size)]
        if PY2:
            self.assertItemsEqual(vars_, ref)
        else:
            self.assertCountEqual(vars_, ref)

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
        ref = [(x.data, size), (y.data, size)]
        if PY2:
            self.assertItemsEqual(vars_, ref)
        else:
            self.assertCountEqual(vars_, ref)

    def test_sum_entries(self):
        """Test sum entries op.
        """
        size = (5, 5)
        x = create_var(size)
        expr = sum_entries(x)
        self.assertEqual(expr.size, (1, 1))
        self.assertEqual(len(expr.args), 1)
        self.assertEqual(expr.type, lo.SUM_ENTRIES)
