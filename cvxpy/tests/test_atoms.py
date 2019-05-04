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

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.transforms.partial_optimize import partial_optimize
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.constants import Parameter, Constant
import numpy as np
from cvxpy import Problem, Minimize
from cvxpy.tests.base_test import BaseTest
import scipy.sparse as sp


class TestAtoms(BaseTest):
    """ Unit tests for the atoms module. """

    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_add_expr_copy(self):
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

    # Test the norm_inf class.
    def test_norm_inf(self):
        exp = self.x+self.y
        atom = cp.norm_inf(exp)
        # self.assertEqual(atom.name(), "norm_inf(x + y)")
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        assert atom.is_convex()
        assert (-atom).is_concave()
        self.assertEqual(cp.norm_inf(atom).curvature, s.CONVEX)
        self.assertEqual(cp.norm_inf(-atom).curvature, s.CONVEX)

    # Test the norm1 class.
    def test_norm1(self):
        exp = self.x+self.y
        atom = cp.norm1(exp)
        # self.assertEqual(atom.name(), "norm1(x + y)")
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(cp.norm1(atom).curvature, s.CONVEX)
        self.assertEqual(cp.norm1(-atom).curvature, s.CONVEX)

    def test_quad_form(self):
        """Test quad_form atom.
        """
        P = Parameter((2, 2))
        expr = cp.quad_form(self.x, P)
        assert not expr.is_dcp()

    # Test the power class.
    def test_power(self):
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

        with self.assertRaises(Exception) as cm:
            cp.power(-1, 3).value
        self.assertEqual(str(cm.exception),
                         "power(x, 3.0) cannot be applied to negative values.")

        with self.assertRaises(Exception) as cm:
            cp.power(0, -1).value
        self.assertEqual(str(cm.exception),
                         "power(x, -1.0) cannot be applied to negative or zero values.")

    # Test the geo_mean class.
    def test_geo_mean(self):
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

    # Test the harmonic_mean class.
    def test_harmonic_mean(self):
        atom = cp.harmonic_mean(self.x)
        self.assertEqual(atom.shape, tuple())
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.NONNEG)

    # Test the pnorm class.
    def test_pnorm(self):
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

    def test_matrix_norms(self):
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

    def test_quad_over_lin(self):
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

    def test_elemwise_arg_count(self):
        """Test arg count for max and min variants.
        """
        with self.assertRaises(Exception) as cm:
            cp.maximum(1)
        self.assertTrue(str(cm.exception) in (
            "__init__() takes at least 3 arguments (2 given)",
            "__init__() missing 1 required positional argument: 'arg2'"))

        with self.assertRaises(Exception) as cm:
            cp.minimum(1)
        self.assertTrue(str(cm.exception) in (
            "__init__() takes at least 3 arguments (2 given)",
            "__init__() missing 1 required positional argument: 'arg2'"))

    def test_matrix_frac(self):
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

    def test_max(self):
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
        self.assertEqual(str(cm.exception),
                         "Invalid argument for axis.")

    def test_min(self):
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
        self.assertEqual(str(cm.exception),
                         "Invalid argument for axis.")

    # Test sign logic for maximum.
    def test_maximum_sign(self):
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
    def test_minimum_sign(self):
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

    def test_sum(self):
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
        self.assertEqual(cp.sum(mat*cp.square(Variable(2))).curvature, s.UNKNOWN)

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

    def test_multiply(self):
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
    def test_vstack(self):
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

    def test_reshape(self):
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

    def test_vec(self):
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

    def test_diag(self):
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

        with self.assertRaises(Exception) as cm:
            cp.diag(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to diag must be a vector or square matrix.")

    def test_trace(self):
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

    def test_log1p(self):
        """Test the log1p atom.
        """
        expr = cp.log1p(1)
        self.assertEqual(expr.sign, s.NONNEG)
        self.assertEqual(expr.curvature, s.CONSTANT)
        self.assertEqual(expr.shape, tuple())
        expr = cp.log1p(-0.5)
        self.assertEqual(expr.sign, s.NONPOS)

    def test_upper_tri(self):
        with self.assertRaises(Exception) as cm:
            cp.upper_tri(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to upper_tri must be a square matrix.")

    def test_huber(self):
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

    def test_sum_largest(self):
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

    def test_sum_smallest(self):
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

    def test_index(self):
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

    def test_bmat(self):
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

    def test_conv(self):
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

    def test_kron(self):
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
            cp.kron(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "The first argument to kron must be constant.")

    def test_partial_optimize_dcp(self):
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

    def test_partial_optimize_eval_1norm(self):
        """Test the partial_optimize atom.
        """
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p1 = Problem(cp.Minimize(cp.sum(t)), [-t <= xval, xval <= t])
        p1.solve()

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
        g = partial_optimize(p2, [t], [x])
        p3 = Problem(cp.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm using maximize.
        p2 = Problem(cp.Maximize(cp.sum(-t)), [-t <= x, x <= t])
        g = partial_optimize(p2, opt_vars=[t])
        p3 = Problem(cp.Maximize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, -p3.value)

        # Try leaving out args.

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cp.Minimize(cp.sum(t)), [-t <= x, x <= t])
        g = partial_optimize(p2, opt_vars=[t])
        p3 = Problem(cp.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm via partial_optimize.
        g = partial_optimize(p2, dont_opt_vars=[x])
        p3 = Problem(cp.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        with self.assertRaises(Exception) as cm:
            g = partial_optimize(p2)
        self.assertEqual(str(cm.exception),
                         "partial_optimize called with neither opt_vars nor dont_opt_vars.")

        with self.assertRaises(Exception) as cm:
            g = partial_optimize(p2, [], [x])
        self.assertEqual(str(cm.exception),
                         ("If opt_vars and new_opt_vars are both specified, "
                          "they must contain all variables in the problem.")
                         )

    def test_partial_optimize_min_1norm(self):
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(cp.sum(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = partial_optimize(p1, [t], [x])
        p2 = Problem(Minimize(g))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)

    def test_partial_optimize_simple_problem(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_var(self):
        x, y = Variable(boolean=True), Variable(integer=True)

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_constr(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x + cp.exp(y)), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(cp.exp(y)), [x+y >= 3, y >= 4])
        g = partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_params(self):
        """Test partial optimize with parameters.
        """
        x, y = Variable(1), Variable(1)
        gamma = Parameter()

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= gamma, y >= 4, x >= 5])
        gamma.value = 3
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= gamma, y >= 4])
        g = partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_numeric_fn(self):
        x, y = Variable(), Variable()
        xval = 4

        # Solve the (simple) two-stage problem by "combining" the two stages
        # (i.e., by solving a single linear program)
        p1 = Problem(Minimize(y), [xval+y >= 3])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        constr = [y >= -100]
        p2 = Problem(Minimize(y), [x+y >= 3] + constr)
        g = partial_optimize(p2, [y], [x])
        x.value = xval
        y.value = 42
        constr[0].dual_variables[0].value = 42
        result = g.value
        self.assertAlmostEqual(result, p1.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(constr[0].dual_value, 42)

        # No variables optimized over.
        p2 = Problem(Minimize(y), [x+y >= 3])
        g = partial_optimize(p2, [], [x, y])
        x.value = xval
        y.value = 42
        p2.constraints[0].dual_variables[0].value = 42
        result = g.value
        self.assertAlmostEqual(result, y.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(p2.constraints[0].dual_value, 42)

    def test_partial_optimize_stacked(self):
        """Minimize the 1-norm in the usual way
        """
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(cp.sum(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = partial_optimize(p1, [t], [x])
        g2 = partial_optimize(Problem(Minimize(g)), [x])
        p2 = Problem(Minimize(g2))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)

    def test_nonnegative_variable(self):
        """Test the NonNegative Variable class.
        """
        x = Variable(nonneg=True)
        p = Problem(Minimize(5+x), [x >= 3])
        p.solve()
        self.assertAlmostEqual(p.value, 8)
        self.assertAlmostEqual(x.value, 3)

    def test_mixed_norm(self):
        """Test mixed norm.
        """
        y = Variable((5, 5))
        obj = Minimize(cp.mixed_norm(y, "inf", 1))
        prob = Problem(obj, [y == np.ones((5, 5))])
        result = prob.solve()
        self.assertAlmostEqual(result, 5)

    def test_indicator(self):
        x = cp.Variable()
        constraints = [0 <= x, x <= 1]
        expr = cp.transforms.indicator(constraints)
        x.value = .5
        self.assertEqual(expr.value, 0.0)
        x.value = 2
        self.assertEqual(expr.value, np.inf)
