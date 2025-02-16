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

import numpy as np
import scipy.sparse as sp

from cvxpy.lin_ops.lin_op import (
    DENSE_CONST,
    NEG,
    PARAM,
    SCALAR_CONST,
    SPARSE_CONST,
    SUM_ENTRIES,
    VARIABLE,
)
from cvxpy.lin_ops.lin_utils import (
    create_const,
    create_eq,
    create_leq,
    create_param,
    create_var,
    get_expr_vars,
    neg_expr,
    sum_entries,
    sum_expr,
)
from cvxpy.tests.base_test import BaseTest


class test_lin_ops(BaseTest):
    """ Unit tests for the lin_ops module. """

    def test_variables(self) -> None:
        """Test creating a variable.
        """
        var = create_var((5, 4), var_id=1)
        self.assertEqual(var.shape, (5, 4))
        self.assertEqual(var.data, 1)
        self.assertEqual(len(var.args), 0)
        self.assertEqual(var.type, VARIABLE)

    def test_param(self) -> None:
        """Test creating a parameter.
        """
        var = create_param((5, 4))
        self.assertEqual(var.shape, (5, 4))
        self.assertEqual(len(var.args), 0)
        self.assertEqual(var.type, PARAM)

    def test_constant(self) -> None:
        """Test creating a constant.
        """
        # Scalar constant.
        shape = (1, 1)
        mat = create_const(1.0, shape)
        self.assertEqual(mat.shape, shape)
        self.assertEqual(len(mat.args), 0)
        self.assertEqual(mat.type, SCALAR_CONST)
        assert mat.data == 1.0

        # Dense matrix constant.
        shape = (5, 4)
        mat = create_const(np.ones(shape), shape)
        self.assertEqual(mat.shape, shape)
        self.assertEqual(len(mat.args), 0)
        self.assertEqual(mat.type, DENSE_CONST)
        assert (mat.data == np.ones(shape)).all()

        # Sparse matrix constant.
        shape = (5, 5)
        mat = create_const(sp.eye_array(5), shape, sparse=True)
        self.assertEqual(mat.shape, shape)
        self.assertEqual(len(mat.args), 0)
        self.assertEqual(mat.type, SPARSE_CONST)
        assert (mat.data.todense() == np.eye(5)).all()

    def test_add_expr(self) -> None:
        """Test adding lin expr.
        """
        shape = (5, 4)
        x = create_var(shape)
        y = create_var(shape)
        # Expanding dict.
        add_expr = sum_expr([x, y])
        self.assertEqual(add_expr.shape, shape)
        assert len(add_expr.args) == 2

    def test_get_vars(self) -> None:
        """Test getting vars from an expression.
        """
        shape = (5, 4)
        x = create_var(shape)
        y = create_var(shape)
        A = create_const(np.ones(shape), shape)
        # Expanding dict.
        add_expr = sum_expr([x, y, A])
        vars_ = get_expr_vars(add_expr)
        ref = [(x.data, shape), (y.data, shape)]
        self.assertCountEqual(vars_, ref)

    def test_neg_expr(self) -> None:
        """Test negating an expression.
        """
        shape = (5, 4)
        var = create_var(shape)
        expr = neg_expr(var)
        assert len(expr.args) == 1
        self.assertEqual(expr.shape, shape)
        self.assertEqual(expr.type, NEG)

    def test_eq_constr(self) -> None:
        """Test creating an equality constraint.
        """
        shape = (5, 5)
        x = create_var(shape)
        y = create_var(shape)
        lh_expr = sum_expr([x, y])
        value = np.ones(shape)
        rh_expr = create_const(value, shape)
        constr = create_eq(lh_expr, rh_expr)
        self.assertEqual(constr.shape, shape)
        vars_ = get_expr_vars(constr.expr)
        ref = [(x.data, shape), (y.data, shape)]
        self.assertCountEqual(vars_, ref)

    def test_leq_constr(self) -> None:
        """Test creating a less than or equal constraint.
        """
        shape = (5, 5)
        x = create_var(shape)
        y = create_var(shape)
        lh_expr = sum_expr([x, y])
        value = np.ones(shape)
        rh_expr = create_const(value, shape)
        constr = create_leq(lh_expr, rh_expr)
        self.assertEqual(constr.shape, shape)
        vars_ = get_expr_vars(constr.expr)
        ref = [(x.data, shape), (y.data, shape)]
        self.assertCountEqual(vars_, ref)

    def test_sum(self) -> None:
        """Test sum entries op.
        """
        shape = (5, 5)
        x = create_var(shape)
        expr = sum_entries(x, (1, 1))
        self.assertEqual(expr.shape, (1, 1))
        self.assertEqual(len(expr.args), 1)
        self.assertEqual(expr.type, SUM_ENTRIES)
