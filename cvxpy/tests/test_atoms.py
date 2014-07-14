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

from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable
from cvxpy.expressions.constants import Parameter
import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf
import numpy as np
import unittest

class TestAtoms(unittest.TestCase):
    """ Unit tests for the atoms module. """
    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Test the norm wrapper.
    def test_norm(self):
        with self.assertRaises(Exception) as cm:
            norm(self.C, 3)
        self.assertEqual(str(cm.exception),
            "Invalid value 3 for p.")

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
        self.assertEquals(atom.size, (1,1))
        self.assertEquals(atom.curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(norm2(atom).curvature, u.Curvature.CONVEX_KEY)
        self.assertEquals(norm2(-atom).curvature, u.Curvature.CONVEX_KEY)

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
        self.assertEqual(str(cm.exception),
            "__init__() takes at least 3 arguments (2 given)")

        with self.assertRaises(Exception) as cm:
            min_elemwise(1)
        self.assertEqual(str(cm.exception),
            "__init__() takes at least 3 arguments (2 given)")

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
