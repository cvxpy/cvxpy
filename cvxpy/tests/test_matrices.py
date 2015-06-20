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
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import *
from cvxpy.expressions.constants import *
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrix_utilities as intf
import numpy
import cvxopt
import scipy
import scipy.sparse as sp
import unittest

class TestMatrices(unittest.TestCase):
    """ Unit tests for testing different forms of matrices as constants. """
    def assertExpression(self, expr, size):
        """Asserts that expr is an Expression with dimension size.
        """
        assert isinstance(expr, Expression) or isinstance(expr, Constraint)
        self.assertEquals(expr.size, size)

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Test numpy arrays
    def test_numpy_arrays(self):
        # Vector
        v = numpy.arange(2).reshape((2,1))
        self.assertExpression(self.x + v, (2,1))
        self.assertExpression(v + self.x, (2,1))
        self.assertExpression(self.x - v, (2,1))
        self.assertExpression(v - self.x, (2,1))
        self.assertExpression(self.x <= v, (2,1))
        self.assertExpression(v <= self.x, (2,1))
        self.assertExpression(self.x == v, (2,1))
        self.assertExpression(v == self.x, (2,1))
        # Matrix
        A = numpy.arange(8).reshape((4,2))
        self.assertExpression(A*self.x, (4,1))
        # PSD inequalities.
        A = numpy.ones((2,2))
        self.assertExpression(A << self.A, (2,2))
        self.assertExpression(A >> self.A, (2,2))

    # Test numpy matrices
    def test_numpy_matrices(self):
        # Vector
        v = numpy.matrix( numpy.arange(2).reshape((2,1)) )
        self.assertExpression(self.x + v, (2,1))
        self.assertExpression(v + v + self.x, (2,1))
        self.assertExpression(self.x - v, (2,1))
        self.assertExpression(v - v - self.x, (2,1))
        self.assertExpression(self.x <= v, (2,1))
        self.assertExpression(v <= self.x, (2,1))
        self.assertExpression(self.x == v, (2,1))
        self.assertExpression(v == self.x, (2,1))
        # Matrix
        A = numpy.matrix( numpy.arange(8).reshape((4,2)) )
        self.assertExpression(A*self.x, (4,1))
        self.assertExpression( (A.T*A) * self.x, (2,1))
        # PSD inequalities.
        A = numpy.matrix(numpy.ones((2,2)))
        self.assertExpression(A << self.A, (2,2))
        self.assertExpression(A >> self.A, (2,2))

    def test_numpy_scalars(self):
        """Test numpy scalars."""
        v = numpy.float64(2.0)
        self.assertExpression(self.x + v, (2,1))
        self.assertExpression(v + self.x, (2,1))
        self.assertExpression(v * self.x, (2,1))
        self.assertExpression(self.x - v, (2,1))
        self.assertExpression(v - v - self.x, (2,1))
        self.assertExpression(self.x <= v, (2,1))
        self.assertExpression(v <= self.x, (2,1))
        self.assertExpression(self.x == v, (2,1))
        self.assertExpression(v == self.x, (2,1))
        # PSD inequalities.
        self.assertExpression(v << self.A, (2,2))
        self.assertExpression(v >> self.A, (2,2))

    # def test_cvxopt_matrices(self):
    #     """Test cvxopt dense matrices.
    #     """
    #     # Vector
    #     v = cvxopt.matrix( numpy.arange(2).reshape((2,1)) )
    #     self.assertExpression(self.x + v, (2,1))
    #     self.assertExpression(v + v + self.x, (2,1))
    #     self.assertExpression(self.x - v, (2,1))
    #     self.assertExpression(v - v - self.x, (2,1))
    #     self.assertExpression(self.x <= v, (2,1))
    #     self.assertExpression(v <= self.x, (2,1))
    #     self.assertExpression(self.x == v, (2,1))
    #     self.assertExpression(v == self.x, (2,1))
    #     # Matrix
    #     A = cvxopt.matrix( numpy.arange(8).reshape((4,2)) )
    #     self.assertExpression(A*self.x, (4,1))
    #     self.assertExpression( (A.T*A) * self.x, (2,1))

    # Test cvxopt sparse matrices.
    def test_cvxopt_sparse(self):
        m = 100
        n = 20

        mu = cvxopt.exp( cvxopt.normal(m) )
        F = cvxopt.normal(m, n)
        D = cvxopt.spdiag( cvxopt.uniform(m) )
        x = Variable(m)
        exp = square(norm2(D*x))

    # def test_scipy_sparse(self):
    #     """Test scipy sparse matrices."""
    #     # Constants.
    #     A = numpy.matrix( numpy.arange(8).reshape((4,2)) )
    #     A = sp.csc_matrix(A)
    #     A = sp.eye(2).tocsc()
    #     key = (slice(0, 1, None), slice(None, None, None))
    #     Aidx = intf.index(A, (slice(0, 2, None), slice(None, None, None)))
    #     Aidx = intf.index(Aidx, key)
    #     self.assertEquals(Aidx.shape, (1, 2))
    #     self.assertEqual(Aidx[0,0], 1)
    #     self.assertEqual(Aidx[0,1], 0)

    #     # Linear ops.
    #     var = Variable(4, 2)
    #     A = numpy.matrix( numpy.arange(8).reshape((4,2)) )
    #     A = sp.csc_matrix(A)
    #     B = sp.hstack([A, A])
    #     self.assertExpression(var + A, (4, 2))
    #     self.assertExpression(A + var, (4, 2))
    #     self.assertExpression(B * var, (4, 2))
    #     self.assertExpression(var - A, (4, 2))
    #     self.assertExpression(A - A - var, (4, 2))
    #     # self.assertExpression(var <= A, (4, 2))
    #     # self.assertExpression(A <= var, (4, 2))
    #     # self.assertExpression(var == A, (4, 2))
    #     # self.assertExpression(A == var, (4, 2))
