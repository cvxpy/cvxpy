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

import cvxpy
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

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

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
        # self.assertEquals(atom.name(), "normInf(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        assert atom.is_convex()
        assert (-atom).is_concave()
        self.assertEquals(normInf(atom).curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(normInf(-atom).curvature, u.Curvature.CONVEX_KEY)

    # Test the norm1 class.
    def test_norm1(self):
        exp = self.x+self.y
        atom = norm1(exp)
        # self.assertEquals(atom.name(), "norm1(x + y)")
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(norm1(atom).curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(norm1(-atom).curvature, u.Curvature.CONVEX_KEY)

    # Test the norm2 class.
    def test_norm2(self):
        exp = self.x+self.y
        atom = norm2(exp)
        # self.assertEquals(atom.name(), "norm2(x + y)")
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(norm2(atom).curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(norm2(-atom).curvature, u.Curvature.CONVEX_KEY)

    # Test the power class.
    def test_power(self):
        from fractions import Fraction

        for size in (1, 1), (3, 1), (2, 3):
            x = Variable(*size)
            y = Variable(*size)
            exp = x + y

            for p in 0, 1, 2, 3, 2.7, .67, -1, -2.3, Fraction(4, 5):
                atom = power(exp, p)

                self.assertEquals(atom.size, size)

                if p > 1 or p < 0:
                    self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
                elif p == 1:
                    self.assertEquals(atom.curvature, u.Curvature.AFFINE_KEY)
                elif p == 0:
                    self.assertEquals(atom.curvature, u.Curvature.CONSTANT_KEY)
                else:
                    self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)

                if p != 1:
                    self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

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


    # Test the geo_mean class.
    def test_geo_mean(self):
        atom = geo_mean(self.x)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)
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
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

    # Test the pnorm class.
    def test_pnorm(self):
        atom = pnorm(self.x, p=1.5)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=1)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=2)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p='inf')
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p='Inf')
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=np.inf)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=.5)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=.7)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=-.1)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=-1)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

        atom = pnorm(self.x, p=-1.3)
        self.assertEquals(atom.size, (1, 1))
        self.assertEquals(atom.curvature, u.Curvature.CONCAVE_KEY)
        self.assertEquals(atom.sign, u.Sign.POSITIVE_KEY)

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
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        atom = quad_over_lin(-square(self.x), self.a)
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        atom = quad_over_lin(sqrt(self.x), self.a)
        self.assertEquals(atom.curvature, u.Curvature.UNKNOWN_KEY)
        assert not atom.is_dcp()

        # Test quad_over_lin size validation.
        with self.assertRaises(Exception) as cm:
            quad_over_lin(self.x, self.x)
        self.assertEqual(str(cm.exception),
            "The second argument to quad_over_lin must be a scalar.")

    def test_elemwise_arg_count(self):
        """Test arg count for max and min variants.
        """
        with self.assertRaises(Exception) as cm:
            max_elemwise(1)
        self.assertTrue(str(cm.exception) in (
            "__init__() takes at least 3 arguments (2 given)",
            "__init__() missing 1 required positional argument: 'arg2'"))

        with self.assertRaises(Exception) as cm:
            min_elemwise(1)
        self.assertTrue(str(cm.exception) in (
            "__init__() takes at least 3 arguments (2 given)",
            "__init__() missing 1 required positional argument: 'arg2'"))

    def test_matrix_frac(self):
        """Test for the matrix_frac atom.
        """
        atom = matrix_frac(self.x, self.A)
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        # Test matrix_frac size validation.
        with self.assertRaises(Exception) as cm:
            matrix_frac(self.x, self.C)
        self.assertEqual(str(cm.exception),
            "The second argument to matrix_frac must be a square matrix.")

        with self.assertRaises(Exception) as cm:
            matrix_frac(self.A, self.A)
        self.assertEqual(str(cm.exception),
            "The first argument to matrix_frac must be a column vector.")

        with self.assertRaises(Exception) as cm:
            matrix_frac(Variable(3), self.A)
        self.assertEqual(str(cm.exception),
            "The arguments to matrix_frac have incompatible dimensions.")

    def test_max_entries_sign(self):
        """Test sign for max_entries.
        """
        # One arg.
        self.assertEquals(max_entries(1).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(max_entries(-2).sign, u.Sign.NEGATIVE_KEY)
        self.assertEquals(max_entries(Variable()).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(max_entries(0).sign, u.Sign.ZERO_KEY)

    def test_min_entries_sign(self):
        """Test sign for min_entries.
        """
        # One arg.
        self.assertEquals(min_entries(1).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(min_entries(-2).sign, u.Sign.NEGATIVE_KEY)
        self.assertEquals(min_entries(Variable()).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(min_entries(0).sign, u.Sign.ZERO_KEY)

    # Test sign logic for max_elemwise.
    def test_max_elemwise_sign(self):
        # Two args.
        self.assertEquals(max_elemwise(1, 2).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(max_elemwise(1, Variable()).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(max_elemwise(1, -2).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(max_elemwise(1, 0).sign, u.Sign.POSITIVE_KEY)

        self.assertEquals(max_elemwise(Variable(), 0).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(max_elemwise(Variable(), Variable()).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(max_elemwise(Variable(), -2).sign, u.Sign.UNKNOWN_KEY)

        self.assertEquals(max_elemwise(0, 0).sign, u.Sign.ZERO_KEY)
        self.assertEquals(max_elemwise(0, -2).sign, u.Sign.ZERO_KEY)

        self.assertEquals(max_elemwise(-3, -2).sign, u.Sign.NEGATIVE_KEY)

        # Many args.
        self.assertEquals(max_elemwise(-2, Variable(), 0, -1, Variable(), 1).sign,
                          u.Sign.POSITIVE_KEY)

        # Promotion.
        self.assertEquals(max_elemwise(1, Variable(2)).sign,
                          u.Sign.POSITIVE_KEY)
        self.assertEquals(max_elemwise(1, Variable(2)).size,
                          (2, 1))

    # Test sign logic for min_elemwise.
    def test_min_elemwise_sign(self):
        # Two args.
        self.assertEquals(min_elemwise(1, 2).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(min_elemwise(1, Variable()).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(min_elemwise(1, -2).sign, u.Sign.NEGATIVE_KEY)
        self.assertEquals(min_elemwise(1, 0).sign, u.Sign.ZERO_KEY)

        self.assertEquals(min_elemwise(Variable(), 0).sign, u.Sign.NEGATIVE_KEY)
        self.assertEquals(min_elemwise(Variable(), Variable()).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(min_elemwise(Variable(), -2).sign, u.Sign.NEGATIVE_KEY)

        self.assertEquals(min_elemwise(0, 0).sign, u.Sign.ZERO_KEY)
        self.assertEquals(min_elemwise(0, -2).sign, u.Sign.NEGATIVE_KEY)

        self.assertEquals(min_elemwise(-3, -2).sign, u.Sign.NEGATIVE_KEY)

        # Many args.
        self.assertEquals(min_elemwise(-2, Variable(), 0, -1, Variable(), 1).sign,
                          u.Sign.NEGATIVE_KEY)

        # Promotion.
        self.assertEquals(min_elemwise(-1, Variable(2)).sign,
                          u.Sign.NEGATIVE_KEY)
        self.assertEquals(min_elemwise(-1, Variable(2)).size,
                          (2, 1))

    def test_sum_entries(self):
        """Test the sum_entries atom.
        """
        self.assertEquals(sum_entries(1).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(sum_entries([1, -1]).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(sum_entries([1, -1]).curvature, u.Curvature.CONSTANT_KEY)
        self.assertEquals(sum_entries(Variable(2)).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(sum_entries(Variable(2)).size, (1, 1))
        self.assertEquals(sum_entries(Variable(2)).curvature, u.Curvature.AFFINE_KEY)
        # Mixed curvature.
        mat = np.mat("1 -1")
        self.assertEquals(sum_entries(mat*square(Variable(2))).curvature, u.Curvature.UNKNOWN_KEY)


    def test_mul_elemwise(self):
        """Test the mul_elemwise atom.
        """
        self.assertEquals(mul_elemwise([1, -1], self.x).sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(mul_elemwise([1, -1], self.x).curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(mul_elemwise([1, -1], self.x).size, (2, 1))
        pos_param = Parameter(2, sign="positive")
        neg_param = Parameter(2, sign="negative")
        self.assertEquals(mul_elemwise(pos_param, pos_param).sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(mul_elemwise(pos_param, neg_param).sign, u.Sign.NEGATIVE_KEY)
        self.assertEquals(mul_elemwise(neg_param, neg_param).sign, u.Sign.POSITIVE_KEY)

        self.assertEquals(mul_elemwise(neg_param, square(self.x)).curvature, u.Curvature.CONCAVE_KEY)

        # Test promotion.
        self.assertEquals(mul_elemwise([1, -1], 1).size, (2, 1))
        self.assertEquals(mul_elemwise(1, self.C).size, self.C.size)

        with self.assertRaises(Exception) as cm:
            mul_elemwise(self.x, [1, -1])
        self.assertEqual(str(cm.exception),
            "The first argument to mul_elemwise must be constant.")

    # Test the vstack class.
    def test_vstack(self):
        atom = vstack(self.x, self.y, self.x)
        self.assertEquals(atom.name(), "vstack(x, y, x)")
        self.assertEquals(atom.size, (6,1))

        atom = vstack(self.A, self.C, self.B)
        self.assertEquals(atom.name(), "vstack(A, C, B)")
        self.assertEquals(atom.size, (7,2))

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
        self.assertEquals(expr.sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(expr.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(expr.size, (4, 1))

        expr = reshape(expr, 2, 2)
        self.assertEquals(expr.size, (2, 2))

        expr = reshape(square(self.x), 1, 2)
        self.assertEquals(expr.sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(expr.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(expr.size, (1, 2))

        with self.assertRaises(Exception) as cm:
            reshape(self.C, 5, 4)
        self.assertEqual(str(cm.exception),
            "Invalid reshape dimensions (5, 4).")

    def test_vec(self):
        """Test the vec atom.
        """
        expr = vec(self.C)
        self.assertEquals(expr.sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(expr.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(expr.size, (6, 1))

        expr = vec(self.x)
        self.assertEquals(expr.size, (2, 1))

        expr = vec(square(self.a))
        self.assertEquals(expr.sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(expr.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(expr.size, (1, 1))

    def test_diag(self):
        """Test the diag atom.
        """
        expr = diag(self.x)
        self.assertEquals(expr.sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(expr.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(expr.size, (2, 2))

        expr = diag(self.A)
        self.assertEquals(expr.sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(expr.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(expr.size, (2, 1))

        with self.assertRaises(Exception) as cm:
            diag(self.C)
        self.assertEqual(str(cm.exception),
            "Argument to diag must be a vector or square matrix.")

    def test_trace(self):
        """Test the trace atom.
        """
        expr = trace(self.A)
        self.assertEquals(expr.sign, u.Sign.UNKNOWN_KEY)
        self.assertEquals(expr.curvature, u.Curvature.AFFINE_KEY)
        self.assertEquals(expr.size, (1, 1))

        with self.assertRaises(Exception) as cm:
            trace(self.C)
        self.assertEqual(str(cm.exception),
            "Argument to trace must be a square matrix.")

    def test_log1p(self):
        """Test the log1p atom.
        """
        expr = log1p(1)
        self.assertEquals(expr.sign, u.Sign.POSITIVE_KEY)
        self.assertEquals(expr.curvature, u.Curvature.CONSTANT_KEY)
        self.assertEquals(expr.size, (1, 1))
        expr = log1p(-0.5)
        self.assertEquals(expr.sign, u.Sign.NEGATIVE_KEY)

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
            huber(self.x, [1,1])
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
            lambda_sum_smallest(Variable(2,2), 2.4)
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
        v_np = np.ones((3,1))
        expr = bmat([[v_np,v_np],[[0,0], [1,2]]])
        self.assertEquals(expr.size, (5,2))
        const = np.bmat([[v_np,v_np],
                        [np.zeros((2,1)), np.mat([1,2]).T]])
        self.assertItemsAlmostEqual(expr.value, const)

    def test_conv(self):
        """Test the conv atom.
        """
        a = np.ones((3,1))
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
            conv([[0,1],[0,1]], self.x)
        self.assertEqual(str(cm.exception),
            "The arguments to conv must resolve to vectors." )

    def test_kron(self):
        """Test the kron atom.
        """
        a = np.ones((3,2))
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

    # Test the partial_optimize atom.
    def test_partial_optimize_eval_1norm(self):
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p1 = Problem(Minimize(sum_entries(t)), [-t<=xval, xval<=t])
        p1.solve()

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(Minimize(sum_entries(t)), [-t<=x, x<=t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        p3 = Problem(Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Try leaving out args.

        # Minimize the 1-norm via partial_optimize.
        g = cvxpy.partial_optimize(p2, opt_vars=[t])
        p3 = Problem(Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm via partial_optimize.
        g = cvxpy.partial_optimize(p2, dont_opt_vars=[x])
        p3 = Problem(Minimize(g), [x == xval])
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
        p1 = Problem(Minimize(sum_entries(t)), [-t<=x, x<=t])

        # Minimize the 1-norm via partial_optimize
        g = cvxpy.partial_optimize(p1, [t], [x])
        p2 = Problem(Minimize(g))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)

    def test_partial_optimize_simple_problem(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y>=3, y>=4, x>=5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y>=3, y>=4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x>=5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_var(self):
        x, y = Bool(1), Int(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y>=3, y>=4, x>=5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y>=3, y>=4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x>=5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_constr(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x + exp(y)), [x+y>=3, y>=4, x>=5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(exp(y)), [x+y>=3, y>=4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x>=5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_params(self):
        """Test partial optimize with parameters.
        """
        x, y = Variable(1), Variable(1)
        gamma = Parameter()
        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y>=gamma, y>=4, x>=5])
        gamma.value = 3
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y>=gamma, y>=4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x>=5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_numeric_fn(self):
        x, y = Variable(1), Variable(1)
        xval = 4

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(y), [xval+y>=3])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y>=3])
        g = cvxpy.partial_optimize(p2, [y], [x])
        x.value = xval
        result = g.value
        self.assertAlmostEqual(result, p1.value)

    def test_partial_optimize_stacked(self):
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(sum_entries(t)), [-t<=x, x<=t])

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
        p = Problem(Minimize(5+x),[x>=3])
        p.solve()
        self.assertAlmostEqual(p.value,8)
        self.assertAlmostEqual(x.value,3)
