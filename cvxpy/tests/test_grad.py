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

from __future__ import division
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

class TestGrad(BaseTest):
    """ Unit tests for the grad module. """
    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    def test_log(self):
        """Test domain for log.
        """
        expr = log(self.a)
        self.a.value = 2
        self.assertAlmostEquals(expr.grad[self.a], 1.0/2)

        self.a.value = 3
        self.assertAlmostEquals(expr.grad[self.a], 1.0/3)

        self.a.value = -1
        self.assertAlmostEquals(expr.grad[self.a], None)

        expr = log(self.x)
        self.x.value = [3,4]
        val = np.zeros((2,2)) + np.diag([1/3,1/4])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = log(self.x)
        self.x.value = [-1e-9,4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = log(self.A)
        self.A.value = [[1,2], [3,4]]
        val = np.zeros((4,4)) + np.diag([1,1/2,1/3,1/4])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_power(self):
        """Test domain for power.
        """
        expr = sqrt(self.a)
        self.a.value = 2
        self.assertAlmostEquals(expr.grad[self.a], 0.5/np.sqrt(2))

        self.a.value = 3
        self.assertAlmostEquals(expr.grad[self.a], 0.5/np.sqrt(3))

        self.a.value = -1
        self.assertAlmostEquals(expr.grad[self.a], None)

        expr = (self.x)**3
        self.x.value = [3,4]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(),
            np.matrix("27 0; 0 48"))

        expr = (self.x)**3
        self.x.value = [-1e-9,4]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), np.matrix("0 0; 0 48"))

        expr = (self.A)**2
        self.A.value = [[1,-2], [3,4]]
        val = np.zeros((4,4)) + np.diag([2,-4,6,8])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_affine(self):
        """Test grad for affine atoms.
        """
        expr = -self.a
        self.a.value = 2
        self.assertAlmostEquals(expr.grad[self.a], -1)

        expr = -(self.x)
        self.x.value = [3,4]
        val = np.zeros((2,2)) - np.diag([1,1])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = -(self.A)
        self.A.value = [[1,2], [3,4]]
        val = np.zeros((4,4)) - np.diag([1,1,1,1])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

        expr = self.A[0,1]
        self.A.value = [[1,2], [3,4]]
        val = np.zeros((4,1))
        val[2] = 1
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

        z = Variable(3)
        expr = vstack(self.x, z)
        self.x.value = [1,2]
        z.value = [1,2,3]
        val = np.zeros((2,5))
        val[:,0:2] = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        val = np.zeros((3,5))
        val[:,2:] = np.eye(3)
        self.assertItemsAlmostEqual(expr.grad[z].todense(), val)

    # def test_log_det(self):
    #     """Test domain for log_det.
    #     """
    #     dom = log_det(self.A + np.eye(2)).domain
    #     prob = Problem(Minimize(sum_entries(diag(self.A))), dom)
    #     prob.solve(solver=cvxpy.SCS)
    #     self.assertAlmostEquals(prob.value, -2, places=3)
