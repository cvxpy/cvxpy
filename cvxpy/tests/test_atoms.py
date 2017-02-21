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

import cvxpy
import cvxpy.settings as s
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative, Bool, Int
from cvxpy.expressions.constants import Parameter
import cvxpy.utilities as u
import numpy as np
import unittest
from cvxpy import Problem, Minimize
from cvxpy.tests.base_test import BaseTest


class TestAtoms(BaseTest):
    """ Unit tests for the atoms module. """

    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

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

    # Test the normInf class.
    def test_normInf(self):
        exp = self.x+self.y
        atom = normInf(exp)
        # self.assertEqual(atom.name(), "normInf(x + y)")
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        assert atom.is_convex()
        assert (-atom).is_concave()
        self.assertEqual(normInf(atom).curvature, s.CONVEX)
        self.assertEqual(normInf(-atom).curvature, s.CONVEX)

    # Test the norm1 class.
    def test_norm1(self):
        exp = self.x+self.y
        atom = norm1(exp)
        # self.assertEqual(atom.name(), "norm1(x + y)")
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(norm1(atom).curvature, s.CONVEX)
        self.assertEqual(norm1(-atom).curvature, s.CONVEX)

    # Test the norm2 class.
    def test_norm2(self):
        exp = self.x+self.y
        atom = norm2(exp)
        # self.assertEqual(atom.name(), "norm2(x + y)")
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(norm2(atom).curvature, s.CONVEX)
        self.assertEqual(norm2(-atom).curvature, s.CONVEX)

        # Test with axis arg.
        expr = norm(self.A, 2, axis=0)
        self.assertEqual(expr.size, (1, 2))

    def test_log_sum_exp(self):
        """Test log_sum_exp"""
        atom = log_sum_exp(vstack(-log(self.x), self.y))
        # self.assertEqual(atom.name(), "norm2(x + y)")
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)

        # Test with axis arg.
        expr = log_sum_exp(self.A, axis=0)
        self.assertEqual(expr.size, (1, 2))

    def test_quad_form(self):
        """Test quad_form atom.
        """
        P = Parameter(2,2)
        with self.assertRaises(Exception) as cm:
            quad_form(self.x, P)
        self.assertEqual(str(cm.exception), "P cannot be a parameter.")

    # Test the power class.
    def test_power(self):
        from fractions import Fraction

        for size in (1, 1), (3, 1), (2, 3):
            x = Variable(*size)
            y = Variable(*size)
            exp = x + y

            for p in 0, 1, 2, 3, 2.7, .67, -1, -2.3, Fraction(4, 5):
                atom = power(exp, p)

                self.assertEqual(atom.size, size)

                if p > 1 or p < 0:
                    self.assertEqual(atom.curvature, s.CONVEX)
                elif p == 1:
                    self.assertEqual(atom.curvature, s.AFFINE)
                elif p == 0:
                    self.assertEqual(atom.curvature, s.CONSTANT)
                else:
                    self.assertEqual(atom.curvature, s.CONCAVE)

                if p != 1:
                    self.assertEqual(atom.sign, s.POSITIVE)

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

        assert power(-1, 2).value == 1

        with self.assertRaises(Exception) as cm:
            power(-1, 3).value
        self.assertEqual(str(cm.exception),
                         "power(x, 3.0) cannot be applied to negative values.")

        with self.assertRaises(Exception) as cm:
            power(0, -1).value
        self.assertEqual(str(cm.exception),
                         "power(x, -1.0) cannot be applied to negative or zero values.")

    # Test the geo_mean class.
    def test_geo_mean(self):
        atom = geo_mean(self.x)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)
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
        atom = harmonic_mean(self.x)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)

    # Test the pnorm class.
    def test_pnorm(self):
        atom = pnorm(self.x, p=1.5)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=1)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=2)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p='inf')
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p='Inf')
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=np.inf)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=.5)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=.7)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=-.1)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=-1)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)

        atom = pnorm(self.x, p=-1.3)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONCAVE)
        self.assertEqual(atom.sign, s.POSITIVE)

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

    def test_quad_over_lin(self):
        # Test quad_over_lin DCP.
        atom = quad_over_lin(square(self.x), self.a)
        self.assertEqual(atom.curvature, s.CONVEX)
        atom = quad_over_lin(-square(self.x), self.a)
        self.assertEqual(atom.curvature, s.CONVEX)
        atom = quad_over_lin(sqrt(self.x), self.a)
        self.assertEqual(atom.curvature, s.UNKNOWN)
        assert not atom.is_dcp()

        # Test quad_over_lin size validation.
        with self.assertRaises(Exception) as cm:
            quad_over_lin(self.x, self.x)
        self.assertEqual(str(cm.exception),
                         "The second argument to quad_over_lin must be a scalar.")

    def test_matrix_frac(self):
        """Test for the matrix_frac atom.
        """
        atom = matrix_frac(self.x, self.A)
        self.assertEqual(atom.size, (1, 1))
        self.assertEqual(atom.curvature, s.CONVEX)
        # Test matrix_frac size validation.
        with self.assertRaises(Exception) as cm:
            matrix_frac(self.x, self.C)
        self.assertEqual(str(cm.exception),
                         "The second argument to matrix_frac must be a square matrix.")

        with self.assertRaises(Exception) as cm:
            matrix_frac(Variable(3), self.A)
        self.assertEqual(str(cm.exception),
                         "The arguments to matrix_frac have incompatible dimensions.")

    def test_max_entries(self):
        """Test max_entries.
        """
        # One arg, test sign.
        self.assertEqual(max_entries(1).sign, s.POSITIVE)
        self.assertEqual(max_entries(-2).sign, s.NEGATIVE)
        self.assertEqual(max_entries(Variable()).sign, s.UNKNOWN)
        self.assertEqual(max_entries(0).sign, s.ZERO)

        # Test with axis argument.
        self.assertEqual(max_entries(Variable(2), axis=0).size, (1, 1))
        self.assertEqual(max_entries(Variable(2), axis=1).size, (2, 1))
        self.assertEqual(max_entries(Variable(2, 3), axis=0).size, (1, 3))
        self.assertEqual(max_entries(Variable(2, 3), axis=1).size, (2, 1))

        # Invalid axis.
        with self.assertRaises(Exception) as cm:
            max_entries(self.x, axis=4)
        self.assertEqual(str(cm.exception),
                         "Invalid argument for axis.")

    def test_min_entries(self):
        """Test min_entries.
        """
        # One arg, test sign.
        self.assertEqual(min_entries(1).sign, s.POSITIVE)
        self.assertEqual(min_entries(-2).sign, s.NEGATIVE)
        self.assertEqual(min_entries(Variable()).sign, s.UNKNOWN)
        self.assertEqual(min_entries(0).sign, s.ZERO)

        # Test with axis argument.
        self.assertEqual(min_entries(Variable(2), axis=0).size, (1, 1))
        self.assertEqual(min_entries(Variable(2), axis=1).size, (2, 1))
        self.assertEqual(min_entries(Variable(2, 3), axis=0).size, (1, 3))
        self.assertEqual(min_entries(Variable(2, 3), axis=1).size, (2, 1))

        # Invalid axis.
        with self.assertRaises(Exception) as cm:
            min_entries(self.x, axis=4)
        self.assertEqual(str(cm.exception),
                         "Invalid argument for axis.")

    # Test sign logic for max_elemwise.
    def test_max_elemwise_sign(self):
        # Two args.
        self.assertEqual(max_elemwise(1, 2).sign, s.POSITIVE)
        self.assertEqual(max_elemwise(1, Variable()).sign, s.POSITIVE)
        self.assertEqual(max_elemwise(1, -2).sign, s.POSITIVE)
        self.assertEqual(max_elemwise(1, 0).sign, s.POSITIVE)

        self.assertEqual(max_elemwise(Variable(), 0).sign, s.POSITIVE)
        self.assertEqual(max_elemwise(Variable(), Variable()).sign, s.UNKNOWN)
        self.assertEqual(max_elemwise(Variable(), -2).sign, s.UNKNOWN)

        self.assertEqual(max_elemwise(0, 0).sign, s.ZERO)
        self.assertEqual(max_elemwise(0, -2).sign, s.ZERO)

        self.assertEqual(max_elemwise(-3, -2).sign, s.NEGATIVE)

        # Many args.
        self.assertEqual(max_elemwise(-2, Variable(), 0, -1, Variable(), 1).sign,
                         s.POSITIVE)

        # Promotion.
        self.assertEqual(max_elemwise(1, Variable(2)).sign,
                         s.POSITIVE)
        self.assertEqual(max_elemwise(1, Variable(2)).size,
                         (2, 1))

    def test_list_input(self):
        """Test *args atoms taking list input.
        """
        with self.assertRaises(Exception) as cm:
            min_elemwise()
        self.assertEqual(str(cm.exception),
                         "min_elemwise requires at least two arguments or a list.")
        with self.assertRaises(Exception) as cm:
            min_elemwise(self.x)
        self.assertEqual(str(cm.exception),
                         "min_elemwise requires at least two arguments or a list.")
        self.assertAlmostEqual(min_elemwise([1, 2]).value, 1)

        with self.assertRaises(Exception) as cm:
            max_elemwise()
        self.assertEqual(str(cm.exception),
                         "max_elemwise requires at least two arguments or a list.")
        with self.assertRaises(Exception) as cm:
            max_elemwise(self.x)
        self.assertEqual(str(cm.exception),
                         "max_elemwise requires at least two arguments or a list.")
        self.assertAlmostEqual(max_elemwise([1, 2]).value, 2)

        # TV
        self.assertAlmostEqual(tv([np.ones((2,2)), np.ones((2,2))]).value, 0.)

        # hstack
        expr = hstack([np.ones((2,2)), np.ones((2,2))])
        self.assertEqual(expr.size, (2, 4))
        self.assertItemsAlmostEqual(expr.value, np.ones((2, 4)))

        # vstack
        expr = vstack([np.ones((2,2)), np.ones((2,2))])
        self.assertEqual(expr.size, (4, 2))
        self.assertItemsAlmostEqual(expr.value, np.ones((4, 2)))

    # Test sign logic for min_elemwise.
    def test_min_elemwise_sign(self):
        # Two args.
        self.assertEqual(min_elemwise(1, 2).sign, s.POSITIVE)
        self.assertEqual(min_elemwise(1, Variable()).sign, s.UNKNOWN)
        self.assertEqual(min_elemwise(1, -2).sign, s.NEGATIVE)
        self.assertEqual(min_elemwise(1, 0).sign, s.ZERO)

        self.assertEqual(min_elemwise(Variable(), 0).sign, s.NEGATIVE)
        self.assertEqual(min_elemwise(Variable(), Variable()).sign, s.UNKNOWN)
        self.assertEqual(min_elemwise(Variable(), -2).sign, s.NEGATIVE)

        self.assertEqual(min_elemwise(0, 0).sign, s.ZERO)
        self.assertEqual(min_elemwise(0, -2).sign, s.NEGATIVE)

        self.assertEqual(min_elemwise(-3, -2).sign, s.NEGATIVE)

        # Many args.
        self.assertEqual(min_elemwise(-2, Variable(), 0, -1, Variable(), 1).sign,
                         s.NEGATIVE)

        # Promotion.
        self.assertEqual(min_elemwise(-1, Variable(2)).sign,
                         s.NEGATIVE)
        self.assertEqual(min_elemwise(-1, Variable(2)).size,
                         (2, 1))

    def test_sum_entries(self):
        """Test the sum_entries atom.
        """
        self.assertEqual(sum_entries(1).sign, s.POSITIVE)
        self.assertEqual(sum_entries([1, -1]).sign, s.UNKNOWN)
        self.assertEqual(sum_entries([1, -1]).curvature, s.CONSTANT)
        self.assertEqual(sum_entries(Variable(2)).sign, s.UNKNOWN)
        self.assertEqual(sum_entries(Variable(2)).size, (1, 1))
        self.assertEqual(sum_entries(Variable(2)).curvature, s.AFFINE)
        # Mixed curvature.
        mat = np.mat("1 -1")
        self.assertEqual(sum_entries(mat*square(Variable(2))).curvature, s.UNKNOWN)

        # Test with axis argument.
        self.assertEqual(sum_entries(Variable(2), axis=0).size, (1, 1))
        self.assertEqual(sum_entries(Variable(2), axis=1).size, (2, 1))
        self.assertEqual(sum_entries(Variable(2, 3), axis=0).size, (1, 3))
        self.assertEqual(sum_entries(Variable(2, 3), axis=1).size, (2, 1))

        # Invalid axis.
        with self.assertRaises(Exception) as cm:
            sum_entries(self.x, axis=4)
        self.assertEqual(str(cm.exception),
                         "Invalid argument for axis.")

    def test_mul_elemwise(self):
        """Test the mul_elemwise atom.
        """
        self.assertEqual(mul_elemwise([1, -1], self.x).sign, s.UNKNOWN)
        self.assertEqual(mul_elemwise([1, -1], self.x).curvature, s.AFFINE)
        self.assertEqual(mul_elemwise([1, -1], self.x).size, (2, 1))
        pos_param = Parameter(2, sign="positive")
        neg_param = Parameter(2, sign="negative")
        self.assertEqual(mul_elemwise(pos_param, pos_param).sign, s.POSITIVE)
        self.assertEqual(mul_elemwise(pos_param, neg_param).sign, s.NEGATIVE)
        self.assertEqual(mul_elemwise(neg_param, neg_param).sign, s.POSITIVE)

        self.assertEqual(mul_elemwise(neg_param, square(self.x)).curvature, s.CONCAVE)

        # Test promotion.
        self.assertEqual(mul_elemwise([1, -1], 1).size, (2, 1))
        self.assertEqual(mul_elemwise(1, self.C).size, self.C.size)

        with self.assertRaises(Exception) as cm:
            mul_elemwise(self.x, [1, -1])
        self.assertEqual(str(cm.exception),
                         "The first argument to mul_elemwise must be constant.")

    # Test the vstack class.
    def test_vstack(self):
        atom = vstack(self.x, self.y, self.x)
        self.assertEqual(atom.name(), "vstack(x, y, x)")
        self.assertEqual(atom.size, (6, 1))

        atom = vstack(self.A, self.C, self.B)
        self.assertEqual(atom.name(), "vstack(A, C, B)")
        self.assertEqual(atom.size, (7, 2))

        entries = []
        for i in range(self.x.size[0]):
            for j in range(self.x.size[1]):
                entries.append(self.x[i, j])
        atom = vstack(*entries)
        # self.assertEqual(atom[1,0].name(), "vstack(x[0,0], x[1,0])[1,0]")

        with self.assertRaises(Exception) as cm:
            vstack(self.C, 1)
        self.assertEqual(str(cm.exception),
                         "All arguments to vstack must have the same number of columns.")

        with self.assertRaises(Exception) as cm:
            vstack()
        self.assertEqual(str(cm.exception),
                         "No arguments given to vstack.")

    def test_reshape(self):
        """Test the reshape class.
        """
        expr = reshape(self.A, 4, 1)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.size, (4, 1))

        expr = reshape(expr, 2, 2)
        self.assertEqual(expr.size, (2, 2))

        expr = reshape(square(self.x), 1, 2)
        self.assertEqual(expr.sign, s.POSITIVE)
        self.assertEqual(expr.curvature, s.CONVEX)
        self.assertEqual(expr.size, (1, 2))

        with self.assertRaises(Exception) as cm:
            reshape(self.C, 5, 4)
        self.assertEqual(str(cm.exception),
                         "Invalid reshape dimensions (5, 4).")

    def test_vec(self):
        """Test the vec atom.
        """
        expr = vec(self.C)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.size, (6, 1))

        expr = vec(self.x)
        self.assertEqual(expr.size, (2, 1))

        expr = vec(square(self.a))
        self.assertEqual(expr.sign, s.POSITIVE)
        self.assertEqual(expr.curvature, s.CONVEX)
        self.assertEqual(expr.size, (1, 1))

    def test_diag(self):
        """Test the diag atom.
        """
        expr = diag(self.x)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.size, (2, 2))

        expr = diag(self.A)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.size, (2, 1))

        expr = diag(self.x.T)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.size, (2, 2))

        with self.assertRaises(Exception) as cm:
            diag(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to diag must be a vector or square matrix.")

    def test_trace(self):
        """Test the trace atom.
        """
        expr = trace(self.A)
        self.assertEqual(expr.sign, s.UNKNOWN)
        self.assertEqual(expr.curvature, s.AFFINE)
        self.assertEqual(expr.size, (1, 1))

        with self.assertRaises(Exception) as cm:
            trace(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to trace must be a square matrix.")

    def test_log1p(self):
        """Test the log1p atom.
        """
        expr = log1p(1)
        self.assertEqual(expr.sign, s.POSITIVE)
        self.assertEqual(expr.curvature, s.CONSTANT)
        self.assertEqual(expr.size, (1, 1))
        expr = log1p(-0.5)
        self.assertEqual(expr.sign, s.NEGATIVE)

    def test_upper_tri(self):
        with self.assertRaises(Exception) as cm:
            upper_tri(self.C)
        self.assertEqual(str(cm.exception),
                         "Argument to upper_tri must be a square matrix.")

    def test_huber(self):
        # Valid.
        huber(self.x, 1)

        with self.assertRaises(Exception) as cm:
            huber(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "M must be a non-negative scalar constant.")

        with self.assertRaises(Exception) as cm:
            huber(self.x, [1, 1])
        self.assertEqual(str(cm.exception),
                         "M must be a non-negative scalar constant.")

        # M parameter.
        M = Parameter(sign="positive")
        # Valid.
        huber(self.x, M)
        M.value = 1
        self.assertAlmostEquals(huber(2, M).value, 3)
        # Invalid.
        M = Parameter(sign="negative")
        with self.assertRaises(Exception) as cm:
            huber(self.x, M)
        self.assertEqual(str(cm.exception),
                         "M must be a non-negative scalar constant.")

        # Test copy with args=None
        atom = huber(self.x, 2)
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
            sum_largest(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        with self.assertRaises(Exception) as cm:
            lambda_sum_largest(self.x, 2.4)
        self.assertEqual(str(cm.exception),
                         "First argument must be a square matrix.")

        with self.assertRaises(Exception) as cm:
            lambda_sum_largest(Variable(2, 2), 2.4)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        # Test copy with args=None
        atom = sum_largest(self.x, 2)
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
        atom = lambda_sum_largest(Variable(2, 2), 2)
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))

    def test_sum_smallest(self):
        """Test the sum_smallest atom and related atoms.
        """
        with self.assertRaises(Exception) as cm:
            sum_smallest(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

        with self.assertRaises(Exception) as cm:
            lambda_sum_smallest(Variable(2, 2), 2.4)
        self.assertEqual(str(cm.exception),
                         "Second argument must be a positive integer.")

    def test_index(self):
        """Test the copy function for index.
        """
        # Test copy with args=None
        size = (5, 4)
        A = Variable(*size)
        atom = A[0:2, 0:1]
        copy = atom.copy()
        self.assertTrue(type(copy) is type(atom))
        # A new object is constructed, so copy.args == atom.args but copy.args
        # is not atom.args.
        self.assertEqual(copy.args, atom.args)
        self.assertFalse(copy.args is atom.args)
        self.assertEqual(copy.get_data(), atom.get_data())
        # Test copy with new args
        B = Variable(4, 5)
        copy = atom.copy(args=[B])
        self.assertTrue(type(copy) is type(atom))
        self.assertTrue(copy.args[0] is B)
        self.assertEqual(copy.get_data(), atom.get_data())

    def test_bmat(self):
        """Test the bmat atom.
        """
        v_np = np.ones((3, 1))
        expr = bmat([[v_np, v_np], [[0, 0], [1, 2]]])
        self.assertEqual(expr.size, (5, 2))
        const = np.bmat([[v_np, v_np],
                         [np.zeros((2, 1)), np.mat([1, 2]).T]])
        self.assertItemsAlmostEqual(expr.value, const)

    def test_conv(self):
        """Test the conv atom.
        """
        a = np.ones((3, 1))
        b = Parameter(2, sign='positive')
        expr = conv(a, b)
        assert expr.is_positive()
        self.assertEqual(expr.size, (4, 1))
        b = Parameter(2, sign='negative')
        expr = conv(a, b)
        assert expr.is_negative()
        with self.assertRaises(Exception) as cm:
            conv(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "The first argument to conv must be constant.")
        with self.assertRaises(Exception) as cm:
            conv([[0, 1], [0, 1]], self.x)
        self.assertEqual(str(cm.exception),
                         "The arguments to conv must resolve to vectors.")

    def test_kron(self):
        """Test the kron atom.
        """
        a = np.ones((3, 2))
        b = Parameter(2, sign='positive')
        expr = kron(a, b)
        assert expr.is_positive()
        self.assertEqual(expr.size, (6, 2))
        b = Parameter(2, sign='negative')
        expr = kron(a, b)
        assert expr.is_negative()
        with self.assertRaises(Exception) as cm:
            kron(self.x, -1)
        self.assertEqual(str(cm.exception),
                         "The first argument to kron must be constant.")

    def test_partial_optimize_dcp(self):
        """Test DCP properties of partial optimize.
        """
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p2 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        self.assertEqual(g.curvature, s.CONVEX)

        p2 = Problem(cvxpy.Maximize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        self.assertEqual(g.curvature, s.CONCAVE)

        p2 = Problem(cvxpy.Maximize(square(t[0])), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        self.assertEquals(g.is_convex(), False)
        self.assertEquals(g.is_concave(), False)

    # Test the partial_optimize atom.
    def test_partial_optimize_eval_1norm(self):
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p1 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= xval, xval <= t])
        p1.solve()

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        p3 = Problem(cvxpy.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm using maximize.
        p2 = Problem(cvxpy.Maximize(sum_entries(-t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, opt_vars=[t])
        p3 = Problem(cvxpy.Maximize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, -p3.value)

        # Try leaving out args.

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, opt_vars=[t])
        p3 = Problem(cvxpy.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm via partial_optimize.
        g = cvxpy.partial_optimize(p2, dont_opt_vars=[x])
        p3 = Problem(cvxpy.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        with self.assertRaises(Exception) as cm:
            g = cvxpy.partial_optimize(p2)
        self.assertEqual(str(cm.exception),
                         "partial_optimize called with neither opt_vars nor dont_opt_vars.")

        with self.assertRaises(Exception) as cm:
            g = cvxpy.partial_optimize(p2, [], [x])
        self.assertEqual(str(cm.exception),
                         ("If opt_vars and new_opt_vars are both specified, "
                          "they must contain all variables in the problem.")
                         )

    def test_partial_optimize_min_1norm(self):
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(sum_entries(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = cvxpy.partial_optimize(p1, [t], [x])
        p2 = Problem(Minimize(g))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)

    def test_partial_optimize_simple_problem(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_var(self):
        x, y = Bool(1), Int(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_constr(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x + exp(y)), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(exp(y)), [x+y >= 3, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_params(self):
        """Test partial optimize with parameters.
        """
        x, y = Variable(1), Variable(1)
        gamma = Parameter()

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= gamma, y >= 4, x >= 5])
        gamma.value = 3
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= gamma, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_numeric_fn(self):
        x, y = Variable(1), Variable(1)
        xval = 4

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(y), [xval+y >= 3])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        constr = [y >= -100]
        p2 = Problem(Minimize(y), [x+y >= 3] + constr)
        g = cvxpy.partial_optimize(p2, [y], [x])
        x.value = xval
        y.value = 42
        constr[0].dual_variable.value = 42
        result = g.value
        self.assertAlmostEqual(result, p1.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(constr[0].dual_value, 42)

        # No variables optimized over.
        p2 = Problem(Minimize(y), [x+y >= 3])
        g = cvxpy.partial_optimize(p2, [], [x, y])
        x.value = xval
        y.value = 42
        p2.constraints[0].dual_variable.value = 42
        result = g.value
        self.assertAlmostEqual(result, y.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(p2.constraints[0].dual_value, 42)

    def test_partial_optimize_stacked(self):
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(sum_entries(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = cvxpy.partial_optimize(p1, [t], [x])
        g2 = cvxpy.partial_optimize(Problem(Minimize(g)), [x])
        p2 = Problem(Minimize(g2))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)

    # Test the NonNegative Variable class.
    def test_nonnegative_variable(self):
        x = NonNegative()
        p = Problem(Minimize(5+x), [x >= 3])
        p.solve()
        self.assertAlmostEqual(p.value, 8)
        self.assertAlmostEqual(x.value, 3)

    def test_change_const(self):
        """Test whether changing an array constant breaks DCP.
        """
        c = np.array([1, 2])
        expr = c.T*square(self.x)
        self.x.value = [1, 1]
        self.assertAlmostEqual(expr.value, 3)
        assert expr.is_dcp()
        c[0] = -1
        self.assertAlmostEqual(expr.value, 3)
        assert expr.is_dcp()
