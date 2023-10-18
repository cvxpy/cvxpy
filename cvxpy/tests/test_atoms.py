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

import unittest

import numpy as np
import pytest
import scipy
import scipy.sparse as sp
import scipy.stats
from numpy import linalg as LA

import cvxpy as cp
import cvxpy.settings as s
from cvxpy import Minimize, Problem
from cvxpy.atoms.errormsg import SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize


class TestAtoms(BaseTest):
    """ Unit tests for the atoms module. """

    def setUp(self) -> None:
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_add_expr_copy(self) -> None:
        """Test the copy function for AddExpresion class.
        """
        atom = self.x + self.y
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with new args
        copy = atom.copy(args=[self.A, self.B])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is self.A)
        self.assertTrue(copy.args[1] is self.B)
        self.assertEqual(copy.get_data(), atom.get_data())

    def test_norm_inf(self) -> None:
        """Test the norm_inf class.
        """
        exp = self.x+self.y
        atom = cp.norm_inf(exp)
        # self.assertEqual(atom.name(), "norm_inf(x + y)")
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        assert atom.is_convex()
        assert (-atom).is_concave()
        self.assertEqual(cp.norm_inf(atom).curvature, s.CONVEX)
        self.assertEqual(cp.norm_inf(-atom).curvature, s.CONVEX)

    def test_norm1(self) -> None:
        """Test the norm1 class.
        """
        exp = self.x+self.y
        atom = cp.norm1(exp)
        # self.assertEqual(atom.name(), "norm1(x + y)")
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(cp.norm1(atom).curvature, s.CONVEX)
        self.assertEqual(cp.norm1(-atom).curvature, s.CONVEX)

    def test_list_input(self) -> None:
        """Test that list input is rejected.
        """
        with self.assertRaises(Exception) as cm:
            cp.max([cp.Variable(), 1])
        self.assertTrue(str(cm.exception) in (
            "The input must be a single CVXPY Expression, not a list. "
            "Combine Expressions using atoms such as bmat, hstack, and vstack."))

        with self.assertRaises(Exception) as cm:
            cp.norm([1, cp.Variable()])
        self.assertTrue(str(cm.exception) in (
            "The input must be a single CVXPY Expression, not a list. "
            "Combine Expressions using atoms such as bmat, hstack, and vstack."))

        x = cp.Variable()
        y = cp.Variable()
        with self.assertRaises(Exception) as cm:
            cp.norm([x, y]) <= 1
        self.assertTrue(str(cm.exception) in (
            "The input must be a single CVXPY Expression, not a list. "
            "Combine Expressions using atoms such as bmat, hstack, and vstack."))

    def test_norm_exceptions(self) -> None:
        """Test that norm exceptions are raised as expected.
        """
        x = cp.Variable(2)
        with self.assertRaises(Exception) as cm:
            cp.norm(x, 'nuc')
        self.assertTrue(str(cm.exception) in (
            "Unsupported norm option nuc for non-matrix."))

    def test_quad_form(self) -> None:
        """Test quad_form atom.
        """
        P = Parameter((2, 2), symmetric=True)
        expr = cp.quad_form(self.x, P)
        assert not expr.is_dcp()

    def test_power(self) -> None:
        """Test the power class.
        """
        from fractions import Fraction

        for shape in [(1, 1), (3, 1), (2, 3)]:
            x = Variable(shape)
            y = Variable(shape)
            exp = x + y

            for p in 0, 1, 2, 3, 2.7, .67, -1, -2.3, Fraction(4, 5):
                atom = cp.power(exp, p)

                self.assertEqual(atom.shape, shape)

                if p > 1 or p < 0:
                    self.assertEqual(atom.curvature, s.CONVEX)
                elif p == 1:
                    self.assertEqual(atom.curvature, s.AFFINE)
                elif p == 0:
                    self.assertEqual(atom.curvature, s.CONSTANT)
                else:
                    self.assertEqual(atom.curvature, s.CONCAVE)

                if p != 1:
                    self.assertEqual(atom.sign, s.NONNEG)

                # Test copy with args=None
                copy = atom.copy()
                self.assertTrue(type(copy) is type(atom))
                # A new object is constructed, so copy.args == atom.args but copy.args
                # is not atom.args.
                self.assertEqual(copy.args, atom.args)
                self.assertFalse(copy.args is atom.args)
                self.assertEqual(copy.get_data(), atom.get_data())
                # Test copy with new args
                copy = atom.copy(args=[self.y])
                self.assertTrue(type(copy) is type(atom))
                self.assertTrue(copy.args[0] is self.y)
                self.assertEqual(copy.get_data(), atom.get_data())

        assert cp.power(-1, 2).value == 1

    # Test the xexp class
    def test_xexp(self) -> None:
        # Test for positive x
        x = Variable(pos=True)
        atom = cp.xexp(x)
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        # Test for negative x
        x = Variable(neg=True)
        atom = cp.xexp(x)
        self.assertNotEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONPOS)

    # Test the geo_mean class.
    def test_geo_mean(self) -> None:
        atom = cp.geo_mean(self.x)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)
        # Test copy with args=None
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with new args
        copy = atom.copy(args=[self.y])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is self.y)
        self.assertEqual(copy.get_data(), atom.get_data())

        # Error message when geo_mean used incorrectly.
        with pytest.raises(
                TypeError, 
                match=SECOND_ARG_SHOULD_NOT_BE_EXPRESSION_ERROR_MESSAGE
            ):
            cp.geo_mean(self.x, self.y)

    # Test the harmonic_mean class.
    def test_harmonic_mean(self) -> None:
        atom = cp.harmonic_mean(self.x)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

    # Test the pnorm class.
    def test_pnorm(self) -> None:
        atom = cp.pnorm(self.x, p=1.5)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=1)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=2)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)
        expr = cp.norm(self.A, 2, axis=0)
        self.assertEqual(expr.shape, (2,))
        expr = cp.norm(self.A, 2, axis=0, keepdims=True)
        self.assertEqual(expr.shape, (1, 2))
        expr = cp.norm(self.A, 2, axis=1, keepdims=True)
        self.assertEqual(expr.shape, (2, 1))

        atom = cp.pnorm(self.x, p='inf')
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p='Inf')
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=np.inf)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=.5)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=.7)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=-.1)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=-1)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

        atom = cp.pnorm(self.x, p=-1.3)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

        # Test copy with args=None
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with new args
        copy = atom.copy(args=[self.y])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is self.y)
        self.assertEqual(copy.get_data(), atom.get_data())

    def test_matrix_norms(self) -> None:
        """
        Matrix 1-norm, 2-norm (sigma_max), infinity-norm,
            Frobenius norm, and nuclear-norm.
        """
        for p in [1, 2, np.inf, 'fro', 'nuc']:
            for var in [self.A, self.C]:
                atom = cp.norm(var, p)
                self.assertEqual(atom.shape, tuple())
                self.assertEqual(atom.curvature, s.CONVEX)
                self.assertEqual(atom.sign, s.NONNEG)
                var.value = np.random.randn(*var.shape)
                self.assertAlmostEqual(atom.value, np.linalg.norm(var.value, ord=p))
        pass

    def test_quad_over_lin(self) -> None:
        # Test quad_over_lin DCP.
        atom = cp.quad_over_lin(cp.square(self.x), self.a)
        self.assertEqual(atom.curvature, s.CONVEX)
        atom = cp.quad_over_lin(-cp.square(self.x), self.a)
        self.assertEqual(atom.curvature, s.CONVEX)
        atom = cp.quad_over_lin(cp.sqrt(self.x), self.a)
        self.assertEqual(atom.curvature, s.UNKNOWN)
        assert not atom.is_dcp()

        # Test quad_over_lin shape validation.
        with self.assertRaises(Exception) as cm:
            cp.quad_over_lin(self.x, self.x)
        self.assertEqual(str(cm.exception),
                         "The second argument to quad_over_lin must be a scalar.")

    def test_elemwise_arg_count(self) -> None:
        """Test arg count for max and min variants.
        """
        error_message = r"__init__\(\) missing 1 required positional argument: 'arg2'"
        with pytest.raises(TypeError, match=error_message):
            cp.maximum(1)

        with pytest.raises(TypeError, match=error_message):
            cp.minimum(1)

    def test_matrix_frac(self) -> None:
        """Test for the matrix_frac atom.
        """
        atom = cp.matrix_frac(self.x, self.A)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        # Test matrix_frac shape validation.
        with self.assertRaises(Exception) as cm:
            cp.matrix_frac(self.x, self.C)
        self.assertEqual(str(cm.exception),
                         "The second argument to matrix_frac must be a square matrix.")

        with self.assertRaises(Exception) as cm:
            cp.matrix_frac(Variable(3), self.A)
        self.assertEqual(str(cm.exception),
                         "The arguments to matrix_frac have incompatible dimensions.")

    def test_max(self) -> None:
        """Test max.
        """
        # One arg, test sign.
        self.assertEqual(cp.max(1).sign, s.NONNEG)
        self.assertEqual(cp.max(-2).sign, s.NONPOS)
        self.assertEqual(cp.max(Variable()).sign, s.UNKNOWN)
        self.assertEqual(cp.max(0).sign, s.ZERO)

        # Test with axis argument.
        self.assertEqual(cp.max(Variable(2), axis=0, keepdims=True).shape, (1,))
        self.assertEqual(cp.max(Variable(2), axis=1).shape, (2,))
        self.assertEqual(cp.max(Variable((2, 3)), axis=0, keepdims=True).shape, (1, 3))
        self.assertEqual(cp.max(Variable((2, 3)), axis=1).shape, (2,))

        # Invalid axis.
        with self.assertRaises(Exception) as cm:
            cp.max(self.x, axis=4)
        self.assertEqual(str(cm.exception), "Invalid argument for axis.")
        with self.assertRaises(ValueError) as cm:
            cp.max(self.x, self.x)  # a common erroneous use-case
        self.assertEqual(str(cm.exception), cp.max.__EXPR_AXIS_ERROR__)

    def test_min(self) -> None:
        """Test min.
        """
        # One arg, test sign.
        self.assertEqual(cp.min(1).sign, s.NONNEG)
        self.assertEqual(cp.min(-2).sign, s.NONPOS)
        self.assertEqual(cp.min(Variable()).sign, s.UNKNOWN)
        self.assertEqual(cp.min(0).sign, s.ZERO)

        # Test with axis argument.
        self.assertEqual(cp.min(Variable(2), axis=0).shape, tuple())
        self.assertEqual(cp.min(Variable(2), axis=1).shape, (2,))
        self.assertEqual(cp.min(Variable((2, 3)), axis=0).shape, (3,))
        self.assertEqual(cp.min(Variable((2, 3)), axis=1).shape, (2,))

        # Invalid axis.
        with self.assertRaises(Exception) as cm:
            cp.min(self.x, axis=4)
        self.assertEqual(str(cm.exception), "Invalid argument for axis.")
        with self.assertRaises(ValueError) as cm:
            cp.min(self.x, self.x)  # a common erroneous use-case
        self.assertEqual(str(cm.exception), cp.min.__EXPR_AXIS_ERROR__)

    # Test sign logic for maximum.
    def test_maximum_sign(self) -> None:
        # Two args.
        self.assertEqual(cp.maximum(1, 2).sign, s.NONNEG)
        self.assertEqual(cp.maximum(1, Variable()).sign, s.NONNEG)
        self.assertEqual(cp.maximum(1, -2).sign, s.NONNEG)
        self.assertEqual(cp.maximum(1, 0).sign, s.NONNEG)

        self.assertEqual(cp.maximum(Variable(), 0).sign, s.NONNEG)
        self.assertEqual(cp.maximum(Variable(), Variable()).sign, s.UNKNOWN)
        self.assertEqual(cp.maximum(Variable(), -2).sign, s.UNKNOWN)

        self.assertEqual(cp.maximum(0, 0).sign, s.ZERO)
        self.assertEqual(cp.maximum(0, -2).sign, s.ZERO)

        self.assertEqual(cp.maximum(-3, -2).sign, s.NONPOS)

        # Many args.
        self.assertEqual(cp.maximum(-2, Variable(), 0, -1, Variable(), 1).sign,
                         s.NONNEG)

        # Promotion.
        self.assertEqual(cp.maximum(1, Variable(2)).sign,
                         s.NONNEG)
        self.assertEqual(cp.maximum(1, Variable(2)).shape,
                         (2,))

    # Test sign logic for minimum.
    def test_minimum_sign(self) -> None:
        # Two args.
        self.assertEqual(cp.minimum(1, 2).sign, s.NONNEG)
        self.assertEqual(cp.minimum(1, Variable()).sign, s.UNKNOWN)
        self.assertEqual(cp.minimum(1, -2).sign, s.NONPOS)
        self.assertEqual(cp.minimum(1, 0).sign, s.ZERO)

        self.assertEqual(cp.minimum(Variable(), 0).sign, s.NONPOS)
        self.assertEqual(cp.minimum(Variable(), Variable()).sign, s.UNKNOWN)
        self.assertEqual(cp.minimum(Variable(), -2).sign, s.NONPOS)

        self.assertEqual(cp.minimum(0, 0).sign, s.ZERO)
        self.assertEqual(cp.minimum(0, -2).sign, s.NONPOS)

        self.assertEqual(cp.minimum(-3, -2).sign, s.NONPOS)

        # Many args.
        self.assertEqual(cp.minimum(-2, Variable(), 0, -1, Variable(), 1).sign,
                         s.NONPOS)

        # Promotion.
        self.assertEqual(cp.minimum(-1, Variable(2)).sign,
                         s.NONPOS)
        self.assertEqual(cp.minimum(-1, Variable(2)).shape,
                         (2,))

    def test_sum(self) -> None:
        """Test the sum atom.
        """
        self.assertEqual(cp.sum(1).sign, s.NONNEG)
        self.assertEqual(cp.sum(Constant([1, -1])).sign, s.UNKNOWN)
        self.assertEqual(cp.sum(Constant([1, -1])).curvature, s.CONSTANT)
        self.assertEqual(cp.sum(Variable(2)).sign, s.UNKNOWN)
        self.assertEqual(cp.sum(Variable(2)).shape, tuple())
        self.assertEqual(cp.sum(Variable(2)).curvature, s.AFFINE)
        self.assertEqual(cp.sum(Variable((2, 1)), keepdims=True).shape, (1, 1))
        # Mixed curvature.
        mat = np.array([[1, -1]])
        self.assertEqual(cp.sum(mat @ cp.square(Variable(2))).curvature, s.UNKNOWN)

        # Test with axis argument.
        self.assertEqual(cp.sum(Variable(2), axis=0).shape, tuple())
        self.assertEqual(cp.sum(Variable(2), axis=1).shape, (2,))
        self.assertEqual(cp.sum(Variable((2, 3)), axis=0, keepdims=True).shape, (1, 3))
        self.assertEqual(cp.sum(Variable((2, 3)), axis=0, keepdims=False).shape, (3,))
        self.assertEqual(cp.sum(Variable((2, 3)), axis=1).shape, (2,))

        # Invalid axis.
        with self.assertRaises(Exception) as cm:
            cp.sum(self.x, axis=4)
        self.assertEqual(str(cm.exception),
                         "Invalid argument for axis.")

        A = sp.eye(3)
        self.assertEqual(cp.sum(A).value, 3)

        A = sp.eye(3)
        self.assertItemsAlmostEqual(cp.sum(A, axis=0).value, [1, 1, 1])

    def test_multiply(self) -> None:
        """Test the multiply atom.
        """
        self.assertEqual(cp.multiply([1, -1], self.x).sign, s.UNKNOWN)
        self.assertEqual(cp.multiply([1, -1], self.x).curvature, s.AFFINE)
        self.assertEqual(cp.multiply([1, -1], self.x).shape, (2,))
        pos_param = Parameter(2, nonneg=True)
        neg_param = Parameter(2, nonpos=True)
        self.assertEqual(cp.multiply(pos_param, pos_param).sign, s.NONNEG)
        self.assertEqual(cp.multiply(pos_param, neg_param).sign, s.NONPOS)
        self.assertEqual(cp.multiply(neg_param, neg_param).sign, s.NONNEG)

        self.assertEqual(cp.multiply(neg_param, cp.square(self.x)).curvature, s.CONCAVE)

        # Test promotion.
        self.assertEqual(cp.multiply([1, -1], 1).shape, (2,))
        self.assertEqual(cp.multiply(1, self.C).shape, self.C.shape)

        self.assertEqual(cp.multiply(self.x, [1, -1]).sign, s.UNKNOWN)
        self.assertEqual(cp.multiply(self.x, [1, -1]).curvature, s.AFFINE)
        self.assertEqual(cp.multiply(self.x, [1, -1]).shape, (2,))

    # Test the vstack class.
    def test_vstack(self) -> None:
        atom = cp.vstack([self.x, self.y, self.x])
        self.assertEqual(atom.name(), "Vstack(x, y, x)")
        self.assertEqual(atom.shape, (3, 2))

        atom = cp.vstack([self.A, self.C, self.B])
        self.assertEqual(atom.name(), "Vstack(A, C, B)")
        self.assertEqual(atom.shape, (7, 2))

        entries = []
        for i in range(self.x.shape[0]):
            entries.append(self.x[i])
        atom = cp.vstack(entries)
        self.assertEqual(atom.shape, (2, 1))
        # self.assertEqual(atom[1,0].name(), "vstack(x[0,0], x[1,0])[1,0]")

        with self.assertRaises(Exception) as cm:
            cp.vstack([self.C, 1])
        self.assertEqual(str(cm.exception),
                         "All the input dimensions except for axis 0 must match exactly.")

        with self.assertRaises(Exception) as cm:
            cp.vstack([self.x, Variable(3)])
        self.assertEqual(str(cm.exception),
                         "All the input dimensions except for axis 0 must match exactly.")

        with self.assertRaises(TypeError) as cm:
            cp.vstack()

        # Test scalars with variables of shape (1,)
        expr = cp.vstack([2, Variable((1,))])
        self.assertEqual(expr.shape, (2, 1))

    def test_reshape(self) -> None:
        """Test the reshape class.
        """
        expr = cp.reshape(self.A, (4, 1))
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, (4, 1))

        expr = cp.reshape(expr, (2, 2))
        self.assertEqual(expr.shape, (2, 2))

        expr = cp.reshape(cp.square(self.x), (1, 2))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertEqual(expr.curvature, s.CONVEX)
        self.assertEqual(expr.shape, (1, 2))

        with self.assertRaises(Exception) as cm:
            cp.reshape(self.C, (5, 4))
        self.assertEqual(str(cm.exception),
                         "Invalid reshape dimensions (5, 4).")

        # Test C-style reshape.
        a = np.arange(10)
        A_np = np.reshape(a, (5, 2), order='C')
        A_cp = cp.reshape(a, (5, 2), order='C')
        self.assertItemsAlmostEqual(A_np, A_cp.value)

        X = cp.Variable((5, 2))
        prob = cp.Problem(cp.Minimize(0), [X == A_cp])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(A_np, X.value)

        a_np = np.reshape(A_np, 10, order='C')
        a_cp = cp.reshape(A_cp, 10, order='C')

        self.assertItemsAlmostEqual(a_np, a_cp.value)

        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(0), [x == a_cp])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(a_np, x.value)

        # Test more complex C-style reshape: matrix to another matrix
        b = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ])
        b_reshaped = b.reshape((2, 6), order='C')
        X = cp.Variable(b.shape)
        X_reshaped = cp.reshape(X, (2, 6), order='C')
        prob = cp.Problem(cp.Minimize(0), [X_reshaped == b_reshaped])
        prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(b_reshaped, X_reshaped.value)
        self.assertItemsAlmostEqual(b, X.value)

    def test_reshape_negative_one(self) -> None:
        """
        Test the reshape class with -1 in the shape.
        """

        expr = cp.Variable((2, 3))
        numpy_expr = np.ones((2, 3))
        shapes = [(-1, 1), (1, -1), (-1, 2), -1, (-1,)]
        expected_shapes = [(6, 1), (1, 6), (3, 2), (6,), (6,)]

        for shape, expected_shape in zip(shapes, expected_shapes):
            expr_reshaped = cp.reshape(expr, shape)
            self.assertEqual(expr_reshaped.shape, expected_shape)

            numpy_expr_reshaped = np.reshape(numpy_expr, shape)
            self.assertEqual(numpy_expr_reshaped.shape, expected_shape)

        with pytest.raises(ValueError, match="Cannot reshape expression"):
            cp.reshape(expr, (8, -1))

        with pytest.raises(AssertionError, match="Only one"):
            cp.reshape(expr, (-1, -1))

        with pytest.raises(ValueError, match="Invalid reshape dimensions"):
            cp.reshape(expr, (-1, 0))

        with pytest.raises(AssertionError, match="Specified dimension must be nonnegative"):
            cp.reshape(expr, (-1, -2))

        A = np.array([[1, 2, 3], [4, 5, 6]])
        A_reshaped = cp.reshape(A, -1, order='C')
        assert np.allclose(A_reshaped.value, A.reshape(-1, order='C'))
        A_reshaped = cp.reshape(A, -1, order='F')
        assert np.allclose(A_reshaped.value, A.reshape(-1, order='F'))

    def test_vec(self) -> None:
        """Test the vec atom.
        """
        expr = cp.vec(self.C)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, (6,))

        expr = cp.vec(self.x)
        self.assertEqual(expr.shape, (2,))

        expr = cp.vec(cp.square(self.a))
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertEqual(expr.curvature, s.CONVEX)
        self.assertEqual(expr.shape, (1,))

    def test_diag(self) -> None:
        """Test the diag atom.
        """
        expr = cp.diag(self.x)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, (2, 2))

        expr = cp.diag(self.A)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, (2,))

        expr = cp.diag(self.x.T)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, (2, 2))

        psd_matrix = np.array([[1, -1], [-1, 1]])
        expr = cp.diag(psd_matrix)
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertEqual(expr.curvature, s.CONSTANT)
        self.assertEqual(expr.shape, (2,))

        with self.assertRaises(Exception) as cm:
            cp.diag(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to diag must be a vector or square matrix.")

        # Test that diag is PSD
        w = np.array([1.0, 2.0])
        expr = cp.diag(w)
        self.assertTrue(expr.is_psd())
        expr = cp.diag(-w)
        self.assertTrue(expr.is_nsd())
        expr = cp.diag(np.array([1, -1]))
        self.assertFalse(expr.is_psd())
        self.assertFalse(expr.is_nsd())

    def test_diag_offset(self) -> None:
        """Test matrix to vector on scalar matrices"""
        test_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        test_vector = np.array([1, 2, 3])
        offsets = [0, 1, -1, 2]
        for offset in offsets:
            a_cp = cp.diag(test_matrix, k=offset)
            a_np = np.diag(test_matrix, k=offset)
            A_cp = cp.diag(test_vector, k=offset)
            A_np = np.diag(test_vector, k=offset)
            self.assertItemsAlmostEqual(a_cp.value, a_np)
            self.assertItemsAlmostEqual(A_cp.value, A_np)

        X = cp.diag(Variable(5), 1)
        self.assertEqual(X.size, 36)

    def test_trace(self) -> None:
        """Test the trace atom.
        """
        expr = cp.trace(self.A)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.shape, tuple())

        with self.assertRaises(Exception) as cm:
            cp.trace(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to trace must be a square matrix.")

    def test_trace_sign_psd(self) -> None:
        """Test sign of trace for psd/nsd inputs.
        """
        X_psd = cp.Variable((2, 2), PSD=True)
        X_nsd = cp.Variable((2, 2), NSD=True)

        psd_trace = cp.trace(X_psd)
        nsd_trace = cp.trace(X_nsd)

        assert psd_trace.is_nonneg()
        assert nsd_trace.is_nonpos()

    def test_log1p(self) -> None:
        """Test the log1p atom.
        """
        expr = cp.log1p(1)
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertEqual(expr.curvature, s.CONSTANT)
        self.assertEqual(expr.shape, tuple())
        expr = cp.log1p(-0.5)
        self.assertEqual(expr.sign, s.NONPOS)

    def test_upper_tri(self) -> None:
        with self.assertRaises(Exception) as cm:
            cp.upper_tri(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to upper_tri must be a square matrix.")

    def test_vec_to_upper_tri(self) -> None:
        x = Variable(shape=(3,))
        X = cp.vec_to_upper_tri(x)
        x.value = np.array([1, 2, 3])
        actual = X.value
        expect = np.array([[1, 2], [0, 3]])
        assert np.allclose(actual, expect)
        y = Variable(shape=(1,))
        y.value = np.array([4])
        Y = cp.vec_to_upper_tri(y, strict=True)
        actual = Y.value
        expect = np.array([[0, 4], [0, 0]])
        assert np.allclose(actual, expect)
        A_expect = np.array([[0, 11, 12, 13],
                             [0, 0, 16, 17],
                             [0, 0, 0, 21],
                             [0, 0, 0, 0]])
        a = np.array([11, 12, 13, 16, 17, 21])
        A_actual = cp.vec_to_upper_tri(a, strict=True).value
        assert np.allclose(A_actual, A_expect)

        with pytest.raises(ValueError, match="must be a triangular number"):
            cp.vec_to_upper_tri(cp.Variable(shape=(4)))

        with pytest.raises(ValueError, match="must be a triangular number"):
            cp.vec_to_upper_tri(cp.Variable(shape=(4)), strict=True)

        with pytest.raises(ValueError, match="must be a vector"):
            cp.vec_to_upper_tri(cp.Variable(shape=(2, 2)))

        # works with row vectors
        assert np.allclose(
            cp.vec_to_upper_tri(np.arange(6)).value, 
            cp.vec_to_upper_tri(np.arange(6).reshape(1, 6)).value
        )

        # works with scalars
        assert np.allclose(cp.vec_to_upper_tri(1, strict=True).value, np.array([[0, 1], [0, 0]]))


    def test_huber(self) -> None:
        # Valid.
        cp.huber(self.x, 1)

        with self.assertRaises(Exception) as cm:
            cp.huber(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "M must be a non-negative scalar constant.")

        with self.assertRaises(Exception) as cm:
            cp.huber(self.x, [1, 1])
        self.assertEqual(str(cm.exception),
                         "M must be a non-negative scalar constant.")

        # M parameter.
        M = Parameter(nonneg=True)
        # Valid.
        cp.huber(self.x, M)
        M.value = 1
        self.assertAlmostEqual(cp.huber(2, M).value, 3)
        # Invalid.
        M = Parameter(nonpos=True)
        with self.assertRaises(Exception) as cm:
            cp.huber(self.x, M)
        self.assertEqual(str(cm.exception),
                         "M must be a non-negative scalar constant.")

        # Test copy with args=None
        atom = cp.huber(self.x, 2)
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        # As get_data() returns a Constant, we have to check the value
        self.assertEqual(copy.get_data()[0].value, atom.get_data()[0].value)
        # Test copy with new args
        copy = atom.copy(args=[self.y])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is self.y)
        self.assertEqual(copy.get_data()[0].value, atom.get_data()[0].value)

    def test_sum_largest(self) -> None:
        """Test the sum_largest atom and related atoms.
        """
        with self.assertRaises(Exception) as cm:
            cp.sum_largest(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        with self.assertRaises(Exception) as cm:
            cp.lambda_sum_largest(self.x, 2.4)
        self.assertEqual(str(cm.exception),
                         "First argument must be a square matrix.")

        with self.assertRaises(Exception) as cm:
            cp.lambda_sum_largest(Variable((2, 2)), 2.4)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        with self.assertRaises(ValueError) as cm:
            cp.lambda_sum_largest([[1, 2], [3, 4]], 2).value
        self.assertEqual(str(cm.exception),
                         "Input matrix was not Hermitian/symmetric.")

        # Test copy with args=None
        atom = cp.sum_largest(self.x, 2)
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with new args
        copy = atom.copy(args=[self.y])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is self.y)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with lambda_sum_largest, which is in fact an AddExpression
        atom = cp.lambda_sum_largest(Variable((2, 2)), 2)
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))

        # Check that sum_largest is PWL so can be canonicalized as a QP.
        atom = cp.sum_largest(self.x, 2)
        assert atom.is_pwl()

    def test_sum_smallest(self) -> None:
        """Test the sum_smallest atom and related atoms.
        """
        with self.assertRaises(Exception) as cm:
            cp.sum_smallest(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        with self.assertRaises(Exception) as cm:
            cp.lambda_sum_smallest(Variable((2, 2)), 2.4)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        # Check that sum_smallest is PWL so can be canonicalized as a QP.
        atom = cp.sum_smallest(self.x, 2)
        assert atom.is_pwl()

    def test_index(self) -> None:
        """Test the copy function for index.
        """
        # Test copy with args=None
        shape = (5, 4)
        A = Variable(shape)
        atom = A[0:2, 0:1]
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with new args
        B = Variable((4, 5))
        copy = atom.copy(args=[B])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is B)
        self.assertEqual(copy.get_data(), atom.get_data())

    def test_bmat(self) -> None:
        """Test the bmat atom.
        """
        v_np = np.ones((3, 1))
        expr = np.vstack([np.hstack([v_np, v_np]),
                          np.hstack([np.zeros((2, 1)),
                                     np.array([[1, 2]]).T])])
        self.assertEqual(expr.shape, (5, 2))
        const = np.vstack([np.hstack([v_np, v_np]),
                           np.hstack([np.zeros((2, 1)),
                                      np.array([[1, 2]]).T])])
        self.assertItemsAlmostEqual(expr, const)

    def test_conv(self) -> None:
        """Test the conv atom.
        """
        a = np.ones((3, 1))
        b = Parameter(2, nonneg=True)
        expr = cp.conv(a, b)
        assert expr.is_nonneg()
        self.assertEqual(expr.shape, (4, 1))
        b = Parameter(2, nonpos=True)
        expr = cp.conv(a, b)
        assert expr.is_nonpos()
        with self.assertRaises(Exception) as cm:
            cp.conv(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "The first argument to conv must be constant.")
        with self.assertRaises(Exception) as cm:
            cp.conv([[0, 1], [0, 1]], self.x)
        self.assertEqual(str(cm.exception),
                         "The arguments to conv must resolve to vectors.")

    def test_kron_expr(self) -> None:
        """Test the kron atom.
        """
        a = np.ones((3, 2))
        b = Parameter((2, 1), nonneg=True)
        expr = cp.kron(a, b)
        assert expr.is_nonneg()
        self.assertEqual(expr.shape, (6, 2))
        b = Parameter((2, 1), nonpos=True)
        expr = cp.kron(a, b)
        assert expr.is_nonpos()
        with self.assertRaises(Exception) as cm:
            cp.kron(self.x, self.x)
        self.assertEqual(str(cm.exception),
                         "At least one argument to kron must be constant.")

    def test_convolve(self) -> None:
        """Test the convolve atom.
        """
        a = np.ones((3,))
        b = Parameter(2, nonneg=True)
        expr = cp.convolve(a, b)
        assert expr.is_nonneg()
        self.assertEqual(expr.shape, (4,))
        b = Parameter(2, nonpos=True)
        expr = cp.convolve(a, b)
        assert expr.is_nonpos()
        with self.assertRaises(Exception) as cm:
            cp.convolve(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "The first argument to conv must be constant.")
        with pytest.raises(ValueError, match="scalar or 1D"):
            cp.convolve([[0, 1], [0, 1]], self.x)

    def test_ptp(self) -> None:
        """Test the ptp atom.
        """
        a = np.array([[10., -10., 3.0], [6., 0., -1.5]])
        expr = cp.ptp(a)
        assert expr.is_nonneg()
        assert expr.shape == ()
        assert np.isclose(expr.value, 20.)

        expr = cp.ptp(a, axis=0)
        assert expr.is_nonneg()
        assert expr.shape == (3,)
        assert np.allclose(expr.value, np.array([4, 10, 4.5]))

        expr = cp.ptp(a, axis=1)
        assert expr.is_nonneg()
        expr.shape == (2,)
        assert np.allclose(expr.value, np.array([20., 7.5]))

        expr = cp.ptp(a, 0, True)
        assert expr.is_nonneg()
        assert expr.shape == (1, 3)
        assert np.allclose(expr.value, np.array([[4, 10, 4.5]]))

        expr = cp.ptp(a, 1, True)
        assert expr.is_nonneg()
        assert expr.shape == (2, 1)
        assert np.allclose(expr.value, np.array([[20.], [7.5]]))

        x = cp.Variable(10)
        expr = cp.ptp(x)
        assert expr.curvature == 'CONVEX'

    def test_stats(self) -> None:
        """Test the mean, std, var atoms.
        """
        a = np.array([[10., 10., 3.0], [6., 0., 1.5]])
        expr_mean = cp.mean(a)
        expr_var = cp.var(a)
        expr_std = cp.std(a)
        assert expr_mean.is_nonneg()
        assert expr_var.is_nonneg()
        assert expr_std.is_nonneg()

        assert np.isclose(a.mean(), expr_mean.value)
        assert np.isclose(a.var(), expr_var.value)
        assert np.isclose(a.std(), expr_std.value)

        for ddof in [0, 1]:
            expr_var = cp.var(a, ddof=ddof)
            expr_std = cp.std(a, ddof=ddof)

            assert np.isclose(a.var(ddof=ddof), expr_var.value)
            assert np.isclose(a.std(ddof=ddof), expr_std.value)

        for axis in [0, 1]:
            for keepdims in [True, False]:
                expr_mean = cp.mean(a, axis=axis, keepdims=keepdims)
                # expr_var = cp.var(a, axis=axis, keepdims=keepdims)
                expr_std = cp.std(a, axis=axis, keepdims=keepdims)

                assert expr_mean.shape == a.mean(axis=axis, keepdims=keepdims).shape
                # assert expr_var.shape == a.var(axis=axis, keepdims=keepdims).shape
                assert expr_std.shape == a.std(axis=axis, keepdims=keepdims).shape

                assert np.allclose(a.mean(axis=axis, keepdims=keepdims), expr_mean.value)
                # assert np.allclose(a.var(axis=axis, keepdims=keepdims), expr_var.value)
                assert np.allclose(a.std(axis=axis, keepdims=keepdims), expr_std.value)

    def test_partial_optimize_dcp(self) -> None:
        """Test DCP properties of partial optimize.
        """
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
        g = partial_optimize(p2, [t], [x])
        self.assertEqual(g.curvature, s.CONVEX)

        p2 = Problem(cp.Maximize(cp.sum(t)), [-t <= x, x <= t])
        g = partial_optimize(p2, [t], [x])
        self.assertEqual(g.curvature, s.CONCAVE)

        p2 = Problem(cp.Maximize(cp.square(t[0])), [-t <= x, x <= t])
        g = partial_optimize(p2, [t], [x])
        self.assertEqual(g.is_convex(), False)
        self.assertEqual(g.is_concave(), False)

    def test_partial_optimize_eval_1norm(self) -> None:
        """Test the partial_optimize atom.
        """
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p1 = Problem(cp.Minimize(cp.sum(t)), [-t <= xval, xval <= t])
        p1.solve(solver='ECOS')

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
        g = partial_optimize(p2, [t], [x], solver='ECOS')
        p3 = Problem(cp.Minimize(g), [x == xval])
        p3.solve(solver='ECOS')
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm using maximize.
        p2 = Problem(cp.Maximize(cp.sum(-t)), [-t <= x, x <= t])
        g = partial_optimize(p2, opt_vars=[t], solver='ECOS')
        p3 = Problem(cp.Maximize(g), [x == xval])
        p3.solve(solver='ECOS')
        self.assertAlmostEqual(p1.value, -p3.value)

        # Try leaving out args.

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
        g = partial_optimize(p2, opt_vars=[t], solver='ECOS')
        p3 = Problem(cp.Minimize(g), [x == xval])
        p3.solve(solver='ECOS')
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm via partial_optimize.
        g = partial_optimize(p2, dont_opt_vars=[x], solver='ECOS')
        p3 = Problem(cp.Minimize(g), [x == xval])
        p3.solve(solver='ECOS')
        self.assertAlmostEqual(p1.value, p3.value)

        with self.assertRaises(Exception) as cm:
            g = partial_optimize(p2, solver='ECOS')
        self.assertEqual(str(cm.exception),
                         "partial_optimize called with neither opt_vars nor dont_opt_vars.")

        with self.assertRaises(Exception) as cm:
            g = partial_optimize(p2, [], [x], solver='ECOS')
        self.assertEqual(str(cm.exception),
                         ("If opt_vars and new_opt_vars are both specified, "
                          "they must contain all variables in the problem.")
                         )

    def test_partial_optimize_min_1norm(self) -> None:
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(cp.sum(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = partial_optimize(p1, [t], [x], solver='ECOS')
        p2 = Problem(Minimize(g))
        p2.solve(solver='ECOS')

        p1.solve(solver='ECOS')
        self.assertAlmostEqual(p1.value, p2.value)

    def test_partial_optimize_simple_problem(self) -> None:
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve(solver=cp.ECOS)

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = partial_optimize(p2, [y], [x], solver='ECOS')
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve(solver=cp.ECOS)
        self.assertAlmostEqual(p1.value, p3.value)

    @unittest.skipUnless(len(INSTALLED_MI_SOLVERS) > 0, 'No mixed-integer solver is installed.')
    def test_partial_optimize_special_var(self) -> None:
        x, y = Variable(boolean=True), Variable(integer=True)

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve(solver=cp.ECOS_BB)

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve(solver=cp.ECOS_BB)
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_constr(self) -> None:
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x + cp.exp(y)), [x+y >= 3, y >= 4, x >= 5])
        p1.solve(solver=cp.SCS, eps=1e-9)

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(cp.exp(y)), [x+y >= 3, y >= 4])
        g = partial_optimize(p2, [y], [x], solver=cp.SCS, eps=1e-9)
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve(solver=cp.SCS, eps=1e-9)
        self.assertAlmostEqual(p1.value, p3.value, places=4)

    def test_partial_optimize_params(self) -> None:
        """Test partial optimize with parameters.
        """
        x, y = Variable(1), Variable(1)
        gamma = Parameter()

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= gamma, y >= 4, x >= 5])
        gamma.value = 3
        p1.solve(solver=cp.SCS, eps=1e-6)

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= gamma, y >= 4])
        g = partial_optimize(p2, [y], [x], solver=cp.SCS, eps=1e-6)
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_numeric_fn(self) -> None:
        x, y = Variable(), Variable()
        xval = 4

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(y), [xval+y >= 3])
        p1.solve(solver=cp.SCS, eps=1e-6)

        # Solve the two-stage problem via partial_optimize
        constr = [y >= -100]
        p2 = Problem(Minimize(y), [x+y >= 3] + constr)
        g = partial_optimize(p2, [y], [x], solver=cp.SCS, eps=1e-6)
        x.value = xval
        y.value = 42
        constr[0].dual_variables[0].value = 42
        result = g.value
        self.assertAlmostEqual(result, p1.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(constr[0].dual_value, 42)

        # No variables optimized over.
        p2 = Problem(Minimize(y), [x+y >= 3])
        g = partial_optimize(p2, [], [x, y], solver=cp.SCS, eps=1e-6)
        x.value = xval
        y.value = 42
        p2.constraints[0].dual_variables[0].value = 42
        result = g.value
        self.assertAlmostEqual(result, y.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(p2.constraints[0].dual_value, 42)

    def test_partial_optimize_stacked(self) -> None:
        """Minimize the 1-norm in the usual way
        """
        dims = 3
        x = Variable(dims, name='x')
        t = Variable(dims, name='t')
        p1 = Problem(Minimize(cp.sum(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = partial_optimize(p1, [t], [x], solver='ECOS')
        g2 = partial_optimize(Problem(Minimize(g)), [x], solver='ECOS')
        p2 = Problem(Minimize(g2))
        p2.solve(solver='ECOS')

        p1.solve(solver='ECOS')
        self.assertAlmostEqual(p1.value, p2.value)

    def test_nonnegative_variable(self) -> None:
        """Test the NonNegative Variable class.
        """
        x = Variable(nonneg=True)
        p = Problem(Minimize(5+x), [x >= 3])
        p.solve(solver=cp.SCS, eps=1e-5)
        self.assertAlmostEqual(p.value, 8)
        self.assertAlmostEqual(x.value, 3)

    def test_mixed_norm(self) -> None:
        """Test mixed norm.
        """
        y = Variable((5, 5))
        obj = Minimize(cp.mixed_norm(y, "inf", 1))
        prob = Problem(obj, [y == np.ones((5, 5))])
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 5)

    def test_mat_norms(self) -> None:
        """Test that norm1 and normInf match definition for matrices.
        """
        A = np.array([[1, 2], [3, 4]])
        print(A)
        X = Variable((2, 2))
        obj = Minimize(cp.norm(X, 1))
        prob = cp.Problem(obj, [X == A])
        result = prob.solve(solver=cp.SCS)
        print(result)
        self.assertAlmostEqual(result, cp.norm(A, 1).value, places=3)

        obj = Minimize(cp.norm(X, np.inf))
        prob = cp.Problem(obj, [X == A])
        result = prob.solve(solver=cp.SCS)
        print(result)
        self.assertAlmostEqual(result, cp.norm(A, np.inf).value, places=3)

    def test_indicator(self) -> None:
        x = cp.Variable()
        constraints = [0 <= x, x <= 1]
        expr = cp.transforms.indicator(constraints)
        x.value = .5
        self.assertEqual(expr.value, 0.0)
        x.value = 2
        self.assertEqual(expr.value, np.inf)

    def test_log_det(self) -> None:
        # test malformed input
        with self.assertRaises(ValueError) as cm:
            cp.log_det([[1, 2], [3, 4]]).value
        self.assertEqual(str(cm.exception),
                         "Input matrix was not Hermitian/symmetric.")

    def test_lambda_max(self) -> None:
        with self.assertRaises(ValueError) as cm:
            cp.lambda_max([[1, 2], [3, 4]]).value
        self.assertEqual(str(cm.exception),
                         "Input matrix was not Hermitian/symmetric.")

    def test_diff(self) -> None:
        """Test the diff atom.
        """
        A = cp.Variable((20, 10))
        B = np.zeros((20, 10))
        self.assertEqual(cp.diff(A, axis=0).shape,
                         np.diff(B, axis=0).shape)
        self.assertEqual(cp.diff(A, axis=1).shape,
                         np.diff(B, axis=1).shape)

        # Issue #1834

        x1 = np.array([[1, 2, 3, 4, 5]])
        x2 = cp.Variable((1, 5), value=x1)

        expr = cp.diff(x1, axis=1)
        self.assertItemsAlmostEqual(expr.value, np.diff(x1, axis=1))
        expr = cp.diff(x2, axis=1)
        self.assertItemsAlmostEqual(expr.value, np.diff(x1, axis=1))

        with pytest.raises(ValueError, match="< k elements"):
            cp.diff(x1, axis=0).value

    def test_log_normcdf(self) -> None:
        self.assertEqual(cp.log_normcdf(self.x).sign, s.NONPOS)
        self.assertEqual(cp.log_normcdf(self.x).curvature, s.CONCAVE)

        for x in range(-4, 5):
            self.assertAlmostEqual(
                np.log(scipy.stats.norm.cdf(x)),
                cp.log_normcdf(x).value,
                places=None,
                delta=1e-2,
            )

        y = Variable((2, 2))
        obj = Minimize(cp.sum(-cp.log_normcdf(y)))
        prob = Problem(obj, [y == 2])
        result = prob.solve(solver=cp.ECOS)
        self.assertAlmostEqual(
            -result, 4 * np.log(scipy.stats.norm.cdf(2)), places=None, delta=1e-2
        )

    def test_scalar_product(self) -> None:
        """Test scalar product.
        """
        p = np.ones((4,))
        v = cp.Variable((4,))

        p = np.ones((4,))
        obj = cp.Minimize(cp.scalar_product(v, p))
        prob = cp.Problem(obj, [v >= 1])
        prob.solve(solver=cp.SCS)
        assert np.allclose(v.value, p)

        # With a parameter.
        p = cp.Parameter((4,))
        v = cp.Variable((4,))

        p.value = np.ones((4,))
        obj = cp.Minimize(cp.scalar_product(v, p))
        prob = cp.Problem(obj, [v >= 1])
        prob.solve(solver=cp.SCS)
        assert np.allclose(v.value, p.value)

    def test_outer(self) -> None:
        """Test the outer atom.
        """
        a = np.ones((3,))
        b = Variable((2,))
        expr = cp.outer(a, b)
        self.assertEqual(expr.shape, (3, 2))

        # Test with parameter
        c = Parameter((2,))
        expr = cp.outer(c, a)
        self.assertEqual(expr.shape, (2, 3))

        d = np.ones((4,))
        expr = cp.outer(a, d)
        true_val = np.outer(a, d)
        assert np.allclose(expr.value, true_val, atol=1e-1)

        # Test with scalars
        assert np.allclose(np.outer(3, 2), cp.outer(3, 2).value)
        assert np.allclose(np.outer(3, d), cp.outer(3, d).value)

        # Test with matrices
        A = np.arange(4).reshape((2, 2))
        np.arange(4, 8).reshape((2, 2))

        with pytest.raises(ValueError, match="x must be a vector"):
            cp.outer(A, d)
        with pytest.raises(ValueError, match="y must be a vector"):
            cp.outer(d, A)

        # allow 2D inputs once row-major flattening is the default
        assert np.allclose(cp.vec(np.array([[1, 2], [3, 4]])).value, np.array([1, 3, 2, 4]))


    def test_conj(self) -> None:
        """Test conj.
        """
        v = cp.Variable((4,))
        obj = cp.Minimize(cp.sum(v))
        prob = cp.Problem(obj, [cp.conj(v) >= 1])
        prob.solve(solver=cp.SCS)
        assert np.allclose(v.value, np.ones((4,)))

    def test_loggamma(self) -> None:
        """Test the approximation of log-gamma.
        """
        # Test evaluation.
        A = np.arange(1, 10)
        A = np.reshape(A, (3, 3))
        true_val = scipy.special.loggamma(A)
        assert np.allclose(cp.loggamma(A).value, true_val, atol=1e-1)

        # Test solving a problem.
        X = cp.Variable((3, 3))
        cost = cp.sum(cp.loggamma(X))
        prob = cp.Problem(cp.Minimize(cost),
                          [X == A])
        result = prob.solve(solver=cp.SCS)
        assert np.isclose(result, true_val.sum(), atol=1e0)

    def test_partial_trace(self) -> None:
        """
        Test partial_trace atom.
        rho_ABC = rho_A \\otimes rho_B \\otimes rho_C.
        Here \\otimes signifies Kronecker product.
        Each rho_i is normalized, i.e. Tr(rho_i) = 1.
        """
        # Set random state.
        np.random.seed(1)

        # Generate five test cases
        rho_A = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
        rho_A /= np.trace(rho_A)
        rho_B = np.random.random((3, 3)) + 1j*np.random.random((3, 3))
        rho_B /= np.trace(rho_B)
        rho_C = np.random.random((2, 2)) + 1j*np.random.random((2, 2))
        rho_C /= np.trace(rho_C)
        rho_AB = np.kron(rho_A, rho_B)
        rho_AC = np.kron(rho_A, rho_C)

        # Construct a cvxpy Variable with value equal to rho_A \otimes rho_B \otimes rho_C.
        temp = np.kron(rho_AB, rho_C)
        rho_ABC = cp.Variable(shape=temp.shape, complex=True)
        rho_ABC.value = temp

        # Try to recover simpler tensors products by taking partial traces of
        # more complicated tensors.
        rho_AB_test = cp.partial_trace(rho_ABC, [4, 3, 2], axis=2)
        rho_AC_test = cp.partial_trace(rho_ABC, [4, 3, 2], axis=1)
        rho_A_test = cp.partial_trace(rho_AB_test, [4, 3], axis=1)
        rho_B_test = cp.partial_trace(rho_AB_test, [4, 3], axis=0)
        rho_C_test = cp.partial_trace(rho_AC_test, [4, 2], axis=0)

        # See if the outputs of partial_trace are correct
        assert np.allclose(rho_AB_test.value, rho_AB)
        assert np.allclose(rho_AC_test.value, rho_AC)
        assert np.allclose(rho_A_test.value, rho_A)
        assert np.allclose(rho_B_test.value, rho_B)
        assert np.allclose(rho_C_test.value, rho_C)

    def test_partial_trace_exceptions(self) -> None:
        """Test exceptions raised by partial trace.
        """
        X = cp.Variable((4, 3))
        with self.assertRaises(ValueError) as cm:
            cp.partial_trace(X, dims=[2, 3], axis=0)
        self.assertEqual(str(cm.exception),
                         "Only supports square matrices.")

        X = cp.Variable((6, 6))
        with self.assertRaises(ValueError) as cm:
            cp.partial_trace(X, dims=[2, 3], axis=-1)
        self.assertEqual(str(cm.exception),
                         "Invalid axis argument, should be between 0 and 2, got -1.")

        X = cp.Variable((6, 6))
        with self.assertRaises(ValueError) as cm:
            cp.partial_trace(X, dims=[2, 4], axis=0)
        self.assertEqual(str(cm.exception),
                         "Dimension of system doesn't correspond to dimension of subsystems.")

    def test_partial_transpose(self) -> None:
        """
        Test out the partial_transpose atom.
        rho_ABC = rho_A \\otimes rho_B \\otimes rho_C.
        Here \\otimes signifies Kronecker product.
        Each rho_i is normalized, i.e. Tr(rho_i) = 1.
        """
        # Set random state.
        np.random.seed(1)

        # Generate three test cases
        rho_A = np.random.random((8, 8)) + 1j*np.random.random((8, 8))
        rho_A /= np.trace(rho_A)
        rho_B = np.random.random((6, 6)) + 1j*np.random.random((6, 6))
        rho_B /= np.trace(rho_B)
        rho_C = np.random.random((4, 4)) + 1j*np.random.random((4, 4))
        rho_C /= np.trace(rho_C)

        rho_TC = np.kron(np.kron(rho_A, rho_B), rho_C.T)
        rho_TB = np.kron(np.kron(rho_A, rho_B.T), rho_C)
        rho_TA = np.kron(np.kron(rho_A.T, rho_B), rho_C)

        # Construct a cvxpy Variable with value equal to rho_A \otimes rho_B \otimes rho_C.
        temp = np.kron(np.kron(rho_A, rho_B), rho_C)
        rho_ABC = cp.Variable(shape=temp.shape, complex=True)
        rho_ABC.value = temp

        # Try to recover simpler tensors products by taking partial transposes of
        # more complicated tensors.
        rho_TC_test = cp.partial_transpose(rho_ABC, [8, 6, 4], axis=2)
        rho_TB_test = cp.partial_transpose(rho_ABC, [8, 6, 4], axis=1)
        rho_TA_test = cp.partial_transpose(rho_ABC, [8, 6, 4], axis=0)

        # See if the outputs of partial_transpose are correct
        assert np.allclose(rho_TC_test.value, rho_TC)
        assert np.allclose(rho_TB_test.value, rho_TB)
        assert np.allclose(rho_TA_test.value, rho_TA)

    def test_partial_transpose_exceptions(self) -> None:
        """Test exceptions raised by partial trace.
        """
        X = cp.Variable((4, 3))
        with self.assertRaises(ValueError) as cm:
            cp.partial_transpose(X, dims=[2, 3], axis=0)
        self.assertEqual(str(cm.exception),
                         "Only supports square matrices.")

        X = cp.Variable((6, 6))
        with self.assertRaises(ValueError) as cm:
            cp.partial_transpose(X, dims=[2, 3], axis=-1)
        self.assertEqual(str(cm.exception),
                         "Invalid axis argument, should be between 0 and 2, got -1.")

        X = cp.Variable((6, 6))
        with self.assertRaises(ValueError) as cm:
            cp.partial_transpose(X, dims=[2, 4], axis=0)
        self.assertEqual(str(cm.exception),
                         "Dimension of system doesn't correspond to dimension of subsystems.")

    def test_log_sum_exp(self) -> None:
        """Test log_sum_exp sign.
        """
        # Test for non-negative x
        x = Variable(nonneg=True)
        atom = cp.log_sum_exp(x)
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.NONNEG)

        # Test for non-positive x
        x = Variable(nonpos=True)
        atom = cp.log_sum_exp(x)
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.UNKNOWN)

    def test_flatten(self) -> None:
        """Test flatten and vec."""
        # Constant argument.
        A = np.arange(10)
        reshaped = np.reshape(A, (2, 5), order='F')
        expr = cp.vec(reshaped, order='F')
        self.assertItemsAlmostEqual(expr.value, A)
        expr = cp.Constant(reshaped).flatten(order='F')
        self.assertItemsAlmostEqual(expr.value, A)

        reshaped = np.reshape(A, (2, 5), order='C')
        expr = cp.vec(reshaped, order='C')
        self.assertItemsAlmostEqual(expr.value, A)
        expr = cp.Constant(reshaped).flatten(order='C')
        self.assertItemsAlmostEqual(expr.value, A)

        reshaped = np.reshape(A, (2, 5), order='F')
        expr = cp.vec(reshaped, order='F')
        self.assertItemsAlmostEqual(expr.value, A)
        expr = cp.Constant(reshaped).flatten()
        self.assertItemsAlmostEqual(expr.value, A)

        # Variable argument.
        x = Variable((2, 5))
        reshaped = np.reshape(A, (2, 5), order='F')
        expr = cp.vec(x, order='F')
        cp.Problem(cp.Minimize(0), [expr == A]).solve()
        self.assertItemsAlmostEqual(x.value, reshaped)
        expr = cp.Constant(A).flatten(order='F')
        cp.Problem(cp.Minimize(0), [expr == A]).solve()
        self.assertItemsAlmostEqual(x.value, reshaped)

        reshaped = np.reshape(A, (2, 5), order='C')
        expr = cp.vec(x, order='C')
        cp.Problem(cp.Minimize(0), [expr == A]).solve()
        self.assertItemsAlmostEqual(x.value, reshaped)
        expr = cp.Constant(A).flatten(order='C')
        cp.Problem(cp.Minimize(0), [expr == A]).solve()
        self.assertItemsAlmostEqual(x.value, reshaped)

        reshaped = np.reshape(A, (2, 5), order='F')
        expr = cp.vec(x)
        cp.Problem(cp.Minimize(0), [expr == A]).solve()
        self.assertItemsAlmostEqual(x.value, reshaped)
        expr = cp.Constant(A).flatten()
        cp.Problem(cp.Minimize(0), [expr == A]).solve()
        self.assertItemsAlmostEqual(x.value, reshaped)

    def test_tr_inv(self) -> None:
        """Test tr_inv atom. """
        T = 5
        # Solves the following SDP problem:
        #           minimize    trace(inv(X))
        #               s.t.    X is PSD
        #                       trace(X)==1

        # Create a symmetric matrix variable.
        X = cp.Variable((T, T), symmetric=True)

        # Define and solve the CVXPY problem.
        # X should be a PSD
        constraints = [X >> 0]
        constraints += [
            cp.trace(X) == 1
        ]
        prob = cp.Problem(cp.Minimize(cp.tr_inv(X)), constraints)
        prob.solve()
        # Check result. The best value is T^2.
        self.assertAlmostEqual(prob.value, T**2)
        X_actual = X.value
        X_expect = np.eye(T) / T
        self.assertItemsAlmostEqual(X_actual, X_expect, places=4)
        # Second SDP problem, given a row full-rank matrix M:
        #           minimize    trace(inv(M * X * M.T))
        #               s.t.    X is PSD
        #                       -1 <= X[i][j] <= 1 for all i,j
        constraints = [X >> 0]
        # n should not be greater than T,
        # because the input should be positive definite.
        n = 4
        M = np.random.randn(n, T)
        constraints += [
            X >= -1,
            X <= 1,
        ]
        prob = cp.Problem(cp.Minimize(cp.tr_inv(M @ X @ M.T)), constraints)
        MM = M @ M.T
        naiveRes = np.sum(LA.eigvalsh(MM) ** -1)
        prob.solve(verbose=True)
        # The optimized result should be smaller than the naive result,
        # where X of the naive result is I.
        self.assertTrue(prob.value < naiveRes)


class TestDotsort(BaseTest):
    """ Unit tests for the dotsort atom. """

    def setUp(self) -> None:
        self.x = cp.Variable(5)

    def test_sum_k_largest_equivalence(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = np.array([1, 1, 1, 0])
        expr = cp.dotsort(self.x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-3:]))

    def test_sum_k_smallest_equivalence(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = np.array([-1, -1, -1, 0])
        expr = -cp.dotsort(self.x, w)
        assert expr.is_concave()
        assert expr.is_decr(0)
        prob = cp.Problem(cp.Maximize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[:3]))

    def test_1D(self):
        x_val = np.array([1, 3, 2, -5, 0])
        w = np.array([-1, 5, 2, 0, 5])
        expr = cp.dotsort(self.x, w)
        assert expr.is_convex()
        assert not expr.is_incr(0)
        assert not expr.is_decr(0)

        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sort(x_val) @ np.sort(w))

    def test_2D(self):
        # X, W are matrix valued
        x = cp.Variable((5, 5))
        x_val = np.arange(25).reshape((5, 5))
        w = np.arange(4).reshape((2, 2))
        w_padded = np.zeros_like(x_val)
        w_padded[:w.shape[0], :w.shape[1]] = w
        expr = cp.dotsort(x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)

        prob = cp.Problem(cp.Minimize(expr), [x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value,
                               np.sort(x_val.flatten()) @ np.sort(w_padded.flatten()))

    def test_0D(self):
        # scalar w
        x_val = np.array([1, 3, 2, -5, 0])
        w = 1
        expr = cp.dotsort(self.x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-1:]))

        # scalar x, w
        x = cp.Variable()
        x_val = np.array([1])
        w = 1
        expr = cp.dotsort(x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)
        prob = cp.Problem(cp.Minimize(expr), [x == x_val])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value, np.sum(np.sort(x_val)[-1:]))

    def test_constant(self):
        x = np.arange(25)
        x_val = np.arange(25).reshape((5, 5))
        w = np.arange(4).reshape((2, 2))
        w_padded = np.zeros_like(x_val)
        w_padded[:w.shape[0], :w.shape[1]] = w
        expr = cp.dotsort(x, w)
        assert expr.is_convex()
        assert expr.is_incr(0)

        prob = cp.Problem(cp.Minimize(expr), [])
        prob.solve()
        self.assertAlmostEqual(prob.objective.value,
                               np.sort(x_val.flatten()) @ np.sort(w_padded.flatten()))

    def test_parameter(self):
        x_val = np.array([1, 3, 2, -5, 0])

        assert cp.dotsort(self.x, cp.Parameter(2, pos=True)).is_incr(0)
        assert cp.dotsort(self.x, cp.Parameter(2, nonneg=True)).is_incr(0)
        assert not cp.dotsort(self.x, cp.Parameter(2, neg=True)).is_incr(0)
        assert cp.dotsort(self.x, cp.Parameter(2, neg=True)).is_decr(0)

        w_p = cp.Parameter(2, value=[1, 0])
        expr = cp.dotsort(self.x, w_p)
        assert not expr.is_incr(0)
        assert not expr.is_decr(0)

        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value,
                               np.sort(x_val) @ np.sort(np.array([1, 0, 0, 0, 0])))
        w_p.value = [-1, -1]
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value,
                               np.sort(x_val) @ np.sort(np.array([-1, -1, 0, 0, 0])))

        # Test parameter affine
        w_p = cp.Parameter(2, value=[1, 0])
        parameter_affine_expression = 2 * w_p
        expr = cp.dotsort(self.x, parameter_affine_expression)
        prob = cp.Problem(cp.Minimize(expr), [self.x == x_val])
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value,
                               np.sort(x_val) @ np.sort(np.array([2, 0, 0, 0, 0])))
        w_p.value = [-1, -1]
        prob.solve(enforce_dpp=True)
        self.assertAlmostEqual(prob.objective.value,
                               np.sort(x_val) @ np.sort(np.array([-2, -2, 0, 0, 0])))

        # Test constant + non-parameter affine
        x_const = np.array([1, 2, 3])
        p = cp.Parameter(value=2)
        p_squared = p ** 2
        expr = cp.dotsort(x_const, p_squared)
        problem = cp.Problem(cp.Minimize(expr))
        problem.solve(enforce_dpp=True)
        self.assertAlmostEqual(expr.value, 2 ** 2 * 3)
        p.value = -1
        problem.solve(enforce_dpp=True)
        self.assertAlmostEqual(expr.value, (-1) ** 2 * 3)

        # Test non-parameter affine w/ with implicit casting to constant
        with pytest.warns(UserWarning,
                          match='You are solving a parameterized problem that is not DPP.'):
            x_val = np.array([1, 2, 3, 4, 5])
            p = cp.Parameter(value=2)
            p_squared = p ** 2
            expr = cp.dotsort(self.x, p_squared)
            problem = cp.Problem(cp.Minimize(expr), [self.x == x_val])
            problem.solve()
            self.assertAlmostEqual(expr.value, 2 ** 2 * 5)
            p.value = -1
            problem.solve()
            self.assertAlmostEqual(expr.value, (-1) ** 2 * 5)

    def test_list(self):
        r = np.array([2, 1, 0, -1, -1])
        w = [1.2, 1.1]
        expr = cp.dotsort(self.x, w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 1)
        self.assertAlmostEqual(self.x.value[:2] @ w, 1)

    def test_composition(self):
        r = np.array([2, 1, 0, -1, -1])
        w = [0.7, 0.8]
        expr = cp.dotsort(cp.exp(self.x), w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 2, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 2)
        self.assertAlmostEqual(np.sort(np.exp(self.x.value))[-2:] @ np.sort(w), 2)

    def test_copy(self):
        # Test copy
        w = np.array([1, 2])
        atom = cp.dotsort(self.x, w)
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertTrue(copy.args[0] is atom.args[0])
        self.assertTrue(copy.args[1] is atom.args[1])

        # Test copy with new args
        copy = atom.copy(args=[self.x, w])
        self.assertFalse(copy.args is atom.args)
        self.assertTrue(copy.args[0] is atom.args[0])
        self.assertFalse(copy.args[1] is atom.args[1])

    def test_non_fixed_x(self):
        r = np.array([2, 1, 0, -1, -1])
        w = np.array([1.2, 1.1])
        expr = cp.dotsort(self.x, w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 1)
        self.assertAlmostEqual(self.x.value[:2] @ w, 1)

        # Test non-fixed x - unordered w
        r = np.array([2, 1, 0, -1, -1])
        w = np.array([1.2, 1.1, 1.3])
        expr = cp.dotsort(self.x, w)
        prob = cp.Problem(cp.Maximize(r @ self.x), [0 <= self.x, expr <= 1, cp.sum(self.x) == 1])
        prob.solve()
        self.assertAlmostEqual(expr.value, 1)
        self.assertAlmostEqual(np.sort(self.x.value)[-3:] @ np.sort(w), 1)

    def test_exceptions(self):
        # len(w) > len(x)
        with self.assertRaises(Exception) as cm:
            cp.dotsort(self.x, [1, 2, 3, 4, 5, 8])
        self.assertEqual(str(cm.exception),
                         "The size of of W must be less or equal to the size of X.")

        # two variable expressions
        with self.assertRaises(Exception) as cm:
            cp.dotsort(self.x, cp.Variable(3))
        self.assertEqual(str(cm.exception),
                         "The W argument must be constant.")

        # swapped arguments
        with self.assertRaises(Exception) as cm:
            cp.dotsort([1, 2, 3], self.x)
        self.assertEqual(str(cm.exception),
                         "The W argument must be constant.")

        # non-dcp composition
        with self.assertRaises(Exception) as cm:
            cp.Problem(cp.Minimize(cp.dotsort(cp.abs(self.x), [-1, 1]))).solve()
        assert "Problem does not follow DCP rules" in str(cm.exception)

        # # non-dpp composition
        p = cp.Parameter(value=2)
        p_squared = p ** 2
        with self.assertRaises(Exception) as cm:
            cp.Problem(cp.Minimize(cp.dotsort(self.x, p_squared))).solve(enforce_dpp=True)
        assert "You are solving a parameterized problem that is not DPP" in str(cm.exception)
