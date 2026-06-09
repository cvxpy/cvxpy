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
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.lin_ops import lin_utils as lu
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

    def test_parameter_and_replacement_helpers(self) -> None:
        x = lu.create_var((2, 2), var_id=10)
        y = lu.create_var((2, 2), var_id=11)
        param = cp.Parameter((2, 2))
        param.value = np.arange(4).reshape((2, 2))
        p_op = lu.create_param((2, 2), param)

        self.assertTrue(lu.is_scalar(lu.create_var(tuple())))
        self.assertTrue(lu.is_const(lu.create_const(1, (1, 1))))
        self.assertIs(lu.get_constr_expr(x, None), x)
        lh_op, rh_op, shape = lu.promote_lin_ops_for_mul(lu.create_var((2,)), lu.create_var((2,)))
        self.assertEqual(lh_op.shape, (1, 2))
        self.assertEqual(rh_op.shape, (2, 1))
        self.assertEqual(shape, (1, 1))
        self.assertEqual(lu.sub_expr(x, y).type, lu.lo.SUM)
        self.assertEqual(lu.rmul_expr(x, p_op, (2, 2)).type, lu.lo.RMUL)
        self.assertEqual(lu.multiply(x, y).type, lu.lo.MUL_ELEM)
        self.assertEqual(lu.kron_r(p_op, x, (4, 4)).type, lu.lo.KRON_R)
        self.assertEqual(lu.kron_l(x, p_op, (4, 4)).type, lu.lo.KRON_L)
        self.assertEqual(lu.div_expr(x, lu.create_const(2, (1, 1))).type, lu.lo.DIV)
        self.assertEqual(lu.promote(lu.create_const(1, (1, 1)), (2, 2)).type, lu.lo.PROMOTE)
        self.assertEqual(lu.broadcast_to([x, y], (2, 2)).type, lu.lo.BROADCAST_TO)
        self.assertEqual(lu.trace(x).type, lu.lo.TRACE)
        self.assertEqual(lu.index(x, (1, 2), (slice(0, 1), slice(None))).type, lu.lo.INDEX)
        self.assertEqual(lu.conv(p_op, x, (3,)).type, lu.lo.CONV)
        self.assertEqual(lu.transpose(lu.create_var((3,))).shape, (3,))
        self.assertEqual(lu.transpose(x).shape, (2, 2))
        self.assertEqual(lu.transpose(lu.create_var((2, 3, 4)), axes=(2, 0, 1)).shape, (4, 2, 3))
        self.assertEqual(lu.reshape(x, (4, 1)).type, lu.lo.RESHAPE)
        self.assertEqual(lu.diag_vec(lu.create_var((2, 1))).shape, (2, 2))
        self.assertEqual(lu.diag_mat(x).shape, (2, 1))
        self.assertEqual(lu.upper_tri(x).shape, (1, 1))
        self.assertEqual(lu.hstack([x, y], (2, 4)).type, lu.lo.HSTACK)
        self.assertEqual(lu.vstack([x, y], (4, 2)).type, lu.lo.VSTACK)
        self.assertEqual(lu.concatenate([x, y], (4, 2), axis=0).type, lu.lo.CONCATENATE)
        geq = lu.create_geq(x, y, constr_id=123)
        self.assertEqual(geq.constr_id, 123)
        self.assertEqual(geq.shape, (2, 2))

        expr = lu.mul_expr(p_op, x, (2, 2))
        self.assertEqual(lu.get_expr_params(expr), [param])
        replacement = lu.create_var((2, 2), var_id=99)
        replaced = lu.replace_new_vars(expr, {10: replacement})
        self.assertIs(replaced.args[0], replacement)

        const_expr = lu.replace_params_with_consts(expr)
        self.assertIn(
            const_expr.data.type,
            {lu.lo.DENSE_CONST, lu.lo.SPARSE_CONST, lu.lo.SCALAR_CONST},
        )
        param.value = None
        with pytest.raises(ValueError, match="missing parameter"):
            lu.check_param_val(param)
