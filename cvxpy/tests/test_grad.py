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

from __future__ import division
import cvxpy
import cvxpy.settings as s
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative, Bool, Int
from cvxpy.expressions.constants import Parameter
import cvxpy.utilities as u
from cvxpy.transforms import linearize
import numpy as np
import unittest
from cvxpy import Problem, Minimize, Maximize
from cvxpy.tests.base_test import BaseTest


class TestGrad(BaseTest):
    """ Unit tests for the grad module. """

    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_affine_prod(self):
        """Test gradient for affine_prod
        """
        expr = affine_prod(self.C, self.A)
        self.C.value = np.matrix([[1, -2], [3, 4], [-1, -3]])
        self.A.value = np.matrix([[3, 2], [-5, 1]])

        self.assertItemsAlmostEqual(expr.grad[self.C].todense(),
                                    np.matrix([[3, 0, 0, 2, 0, 0], [0, 3, 0, 0, 2, 0], [0, 0, 3, 0, 0, 2],
                                               [-5, 0, 0, 1, 0, 0], [0, -5, 0, 0, 1, 0], [0, 0, -5, 0, 0, 1]]))
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(),
                                    np.matrix([[1, 3, -1, 0, 0, 0], [-2, 4, -3, 0, 0, 0],
                                               [0, 0, 0, 1, 3, -1], [0, 0, 0, -2, 4, -3]]))

    def test_pnorm(self):
        """Test gradient for pnorm
        """
        expr = pnorm(self.x, 1)
        self.x.value = [-1, 0]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [-1, 0])

        self.x.value = [0, 10]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [0, 1])

        expr = pnorm(self.x, 2)
        self.x.value = [-3, 4]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), np.matrix([[-3.0/5], [4.0/5]]))

        expr = pnorm(self.x, 0.5)
        self.x.value = [-1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = pnorm(self.x, 0.5)
        self.x.value = [0, 0]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = pnorm(self.x, 2)
        self.x.value = [0, 0]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [0, 0])

        expr = pnorm(self.x, 2, axis=1)
        self.x.value = [1, 2]
        val = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = pnorm(self.A, 2)
        self.A.value = np.matrix([[2, -2], [2, 2]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [0.5, 0.5, -0.5, 0.5])

        expr = pnorm(self.A, 2, axis=0)
        self.A.value = np.matrix([[3, -3], [4, 4]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), np.matrix([[0.6, 0], [0.8, 0], [0, -0.6], [0, 0.8]]))

        expr = pnorm(self.A, 2, axis=1)
        self.A.value = np.matrix([[3, -4], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), np.matrix([[0.6, 0], [0, 0.8], [-0.8, 0], [0, 0.6]]))

        expr = pnorm(self.A, 0.5)
        self.A.value = np.matrix([[3, -4], [4, 3]])
        self.assertAlmostEqual(expr.grad[self.A], None)

    def test_log_sum_exp(self):
        expr = log_sum_exp(self.x)
        self.x.value = [0, 1]
        e = np.exp(1)
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [1.0/(1+e), e/(1+e)])

        expr = log_sum_exp(self.A)
        self.A.value = np.matrix([[0, 1], [-1, 0]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [1.0/(2+e+1.0/e), 1.0/e/(2+e+1.0/e), e/(2+e+1.0/e), 1.0/(2+e+1.0/e)])

        expr = log_sum_exp(self.A, axis=0)
        self.A.value = np.matrix([[0, 1], [-1, 0]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(),
                                    np.transpose(np.matrix([[1.0/(1+1.0/e), 1.0/e/(1+1.0/e), 0, 0], [0, 0, e/(1+e), 1.0/(1+e)]])))

    def test_geo_mean(self):
        """Test gradient for geo_mean
        """
        expr = geo_mean(self.x)
        self.x.value = [1, 2]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [np.sqrt(2)/2, 1.0/2/np.sqrt(2)])

        self.x.value = [0, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = geo_mean(self.x, [1, 0])
        self.x.value = [1, 2]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [1, 0])

        # No exception for single weight.
        self.x.value = [-1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)

    def test_lambda_max(self):
        """Test gradient for lambda_max
        """
        expr = lambda_max(self.A)
        self.A.value = [[2, 0], [0, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [1, 0, 0, 0])

        self.A.value = [[1, 0], [0, 2]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [0, 0, 0, 1])

        self.A.value = [[1, 0], [0, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [0, 0, 0, 1])

    def test_matrix_frac(self):
        """Test gradient for matrix_frac
        """
        expr = matrix_frac(self.A, self.B)
        self.A.value = np.eye(2)
        self.B.value = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [2, 0, 0, 2])
        self.assertItemsAlmostEqual(expr.grad[self.B].todense(), [-1, 0, 0, -1])

        self.B.value = np.zeros((2, 2))
        self.assertAlmostEqual(expr.grad[self.A], None)
        self.assertAlmostEqual(expr.grad[self.B], None)

        expr = matrix_frac(self.x, self.A)
        self.x.value = [2, 3]
        self.A.vaule = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [4, 6])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [-4, -6, -6, -9])

    def test_norm_nuc(self):
        """Test gradient for norm_nuc
        """
        expr = normNuc(self.A)
        self.A.value = [[10, 4], [4, 30]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [1, 0, 0, 1])

    def test_log_det(self):
        """Test gradient for log_det
        """
        expr = log_det(self.A)
        self.A.value = 2*np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), 1.0/2*np.eye(2))

        mat = np.matrix([[1, 2], [3, 5]])
        self.A.value = mat.T*mat
        val = np.linalg.inv(self.A.value).T
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

        self.A.value = np.zeros((2, 2))
        self.assertAlmostEqual(expr.grad[self.A], None)

        self.A.value = -np.matrix([[1, 2], [3, 4]])
        self.assertAlmostEqual(expr.grad[self.A], None)

        K = Variable(8, 8)
        expr = log_det(K[[1,2]][:,[1,2]])
        K.value = np.eye(8)
        val = np.zeros((8,8))
        val[[1,2],[1,2]] = 1
        self.assertItemsAlmostEqual(expr.grad[K].todense(), val)

    def test_quad_over_lin(self):
        """Test gradient for quad_over_lin
        """
        expr = quad_over_lin(self.x, self.a)
        self.x.value = [1, 2]
        self.a.value = 2
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [1, 2])
        self.assertAlmostEqual(expr.grad[self.a], [-1.25])

        self.a.value = 0
        self.assertAlmostEqual(expr.grad[self.x], None)
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = quad_over_lin(self.A, self.a)
        self.A.value = np.eye(2)
        self.a.value = 2
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [1, 0, 0, 1])
        self.assertAlmostEqual(expr.grad[self.a], [-0.5])

        expr = quad_over_lin(self.x, self.a) + quad_over_lin(self.y, self.a)
        self.x.value = [1, 2]
        self.a.value = 2
        self.y.value = [1, 2]
        self.a.value = 2
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [1, 2])
        self.assertItemsAlmostEqual(expr.grad[self.y].todense(), [1, 2])
        self.assertAlmostEqual(expr.grad[self.a], [-2.5])

    def test_max_entries(self):
        """Test gradient for max_entries
        """
        expr = max_entries(self.x)
        self.x.value = [2, 1]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), [1, 0])

        expr = max_entries(self.A)
        self.A.value = np.matrix([[1, 2], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [0, 1, 0, 0])

        expr = max_entries(self.A, axis=0)
        self.A.value = np.matrix([[1, 2], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), np.matrix([[0, 0], [1, 0], [0, 0], [0, 1]]))

        expr = max_entries(self.A, axis=1)
        self.A.value = np.matrix([[1, 2], [4, 3]])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), np.matrix([[0, 0], [0, 1], [1, 0], [0, 0]]))

    def test_sigma_max(self):
        """Test sigma_max.
        """
        expr = sigma_max(self.A)
        self.A.value = [[1, 0], [0, 2]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [0, 0, 0, 1])

        self.A.value = [[1, 0], [0, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [1, 0, 0, 0])

    def test_sum_largest(self):
        """Test sum_largest.
        """
        expr = sum_largest(self.A, 2)

        self.A.value = [[4, 3], [2, 1]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [1, 0, 1, 0])

        self.A.value = [[1, 2], [3, 0.5]]
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), [0, 1, 1, 0])

    def test_abs(self):
        """Test abs.
        """
        expr = abs(self.A)
        self.A.value = [[1, 2], [-1, 0]]
        val = np.zeros((4, 4)) + np.diag([1, 1, -1, 0])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_linearize(self):
        """Test linearize method.
        """
        # Affine.
        expr = (2*self.x - 5)[0]
        self.x.value = [1, 2]
        lin_expr = linearize(expr)
        self.x.value = [55, 22]
        self.assertAlmostEqual(lin_expr.value, expr.value)
        self.x.value = [-1, -5]
        self.assertAlmostEqual(lin_expr.value, expr.value)

        # Convex.
        expr = (self.A)**2 + 5

        with self.assertRaises(Exception) as cm:
            linearize(expr)
        self.assertEqual(str(cm.exception),
                         "Cannot linearize non-affine expression with missing variable values.")

        self.A.value = [[1, 2], [3, 4]]
        lin_expr = linearize(expr)
        manual = expr.value + 2*reshape(diag(vec(self.A)).value*vec(self.A - self.A.value), 2, 2)
        self.assertItemsAlmostEqual(lin_expr.value, expr.value)
        self.A.value = [[-5, -5], [8.2, 4.4]]
        assert (lin_expr.value <= expr.value).all()
        self.assertItemsAlmostEqual(lin_expr.value, manual.value)

        # Concave.
        expr = log(self.x)/2
        self.x.value = [1, 2]
        lin_expr = linearize(expr)
        manual = expr.value + diag(0.5*self.x**-1).value*(self.x - self.x.value)
        self.assertItemsAlmostEqual(lin_expr.value, expr.value)
        self.x.value = [3, 4.4]
        assert (lin_expr.value >= expr.value).all()
        self.assertItemsAlmostEqual(lin_expr.value, manual.value)

    def test_log(self):
        """Test gradient for log.
        """
        expr = log(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 1.0/2)

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 1.0/3)

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = log(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag([1/3, 1/4])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = log(self.x)
        self.x.value = [-1e-9, 4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = log(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([1, 1/2, 1/3, 1/4])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_log1p(self):
        """Test domain for log1p.
        """
        expr = log1p(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 1.0/3)

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 1.0/4)

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = log1p(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag([1/4, 1/5])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = log1p(self.x)
        self.x.value = [-1e-9-1, 4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = log1p(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([1/2, 1/3, 1/4, 1/5])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_entr(self):
        """Test domain for entr.
        """
        expr = entr(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], -np.log(2) - 1)

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], -(np.log(3) + 1))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = entr(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag(-(np.log([3, 4]) + 1))
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = entr(self.x)
        self.x.value = [-1e-9, 4]
        self.assertAlmostEqual(expr.grad[self.x], None)

        expr = entr(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag(-(np.log([1, 2, 3, 4]) + 1))
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_exp(self):
        """Test domain for exp.
        """
        expr = exp(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], np.exp(2))

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], np.exp(3))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], np.exp(-1))

        expr = exp(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([3, 4]))
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = exp(self.x)
        self.x.value = [-1e-9, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([-1e-9, 4]))
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = exp(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag(np.exp([1, 2, 3, 4]))
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_logistic(self):
        """Test domain for logistic.
        """
        expr = logistic(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], np.exp(2)/(1+np.exp(2)))

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], np.exp(3)/(1+np.exp(3)))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], np.exp(-1)/(1+np.exp(-1)))

        expr = logistic(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([3, 4])/(1+np.exp([3, 4])))
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = logistic(self.x)
        self.x.value = [-1e-9, 4]
        val = np.zeros((2, 2)) + np.diag(np.exp([-1e-9, 4])/(1+np.exp([-1e-9, 4])))
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = logistic(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag(np.exp([1, 2, 3, 4])/(1+np.exp([1, 2, 3, 4])))
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_huber(self):
        """Test domain for huber.
        """
        expr = huber(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 2)

        expr = huber(self.a, M=2)
        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 4)

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], -2)

        expr = huber(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) + np.diag([2, 2])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = huber(self.x)
        self.x.value = [-1e-9, 4]
        val = np.zeros((2, 2)) + np.diag([0, 2])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = huber(self.A, M=3)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([2, 4, 6, 6])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

    def test_kl_div(self):
        """Test domain for kl_div.
        """
        b = Variable()
        expr = kl_div(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], np.log(2/4))
        self.assertAlmostEqual(expr.grad[b], 1 - (2/4))

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], None)
        self.assertAlmostEqual(expr.grad[b], None)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], None)
        self.assertAlmostEqual(expr.grad[b], None)

        y = Variable(2)
        expr = kl_div(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, 8]
        val = np.zeros((2, 2)) + np.diag(np.log([3, 4]) - np.log([5, 8]))
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)
        val = np.zeros((2, 2)) + np.diag([1 - 3/5, 1 - 4/8])
        self.assertItemsAlmostEqual(expr.grad[y].todense(), val)

        expr = kl_div(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 2]
        self.assertAlmostEqual(expr.grad[self.x], None)
        self.assertAlmostEqual(expr.grad[y], None)

        expr = kl_div(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3.5, 2.3]]
        div = (self.A.value/self.B.value).A.ravel(order='F')
        val = np.zeros((4, 4)) + np.diag(np.log(div))
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)
        val = np.zeros((4, 4)) + np.diag(1 - div)
        self.assertItemsAlmostEqual(expr.grad[self.B].todense(), val)

    def test_max_elemwise(self):
        """Test domain for max_elemwise.
        """
        b = Variable()
        expr = max_elemwise(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], 0)
        self.assertAlmostEqual(expr.grad[b], 1)

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], 1)
        self.assertAlmostEqual(expr.grad[b], 0)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 0)
        self.assertAlmostEqual(expr.grad[b], 1)

        y = Variable(2)
        expr = max_elemwise(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, -5]
        val = np.zeros((2, 2)) + np.diag([0, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)
        val = np.zeros((2, 2)) + np.diag([1, 0])
        self.assertItemsAlmostEqual(expr.grad[y].todense(), val)

        expr = max_elemwise(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 4]
        val = np.zeros((2, 2)) + np.diag([0, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)
        val = np.zeros((2, 2)) + np.diag([1, 0])
        self.assertItemsAlmostEqual(expr.grad[y].todense(), val)

        expr = max_elemwise(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3, 2.3]]
        div = (self.A.value/self.B.value).A.ravel(order='F')
        val = np.zeros((4, 4)) + np.diag([0, 1, 1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)
        val = np.zeros((4, 4)) + np.diag([1, 0, 0, 0])
        self.assertItemsAlmostEqual(expr.grad[self.B].todense(), val)

    def test_min_elemwise(self):
        """Test domain for min_elemwise.
        """
        b = Variable()
        expr = min_elemwise(self.a, b)
        self.a.value = 2
        b.value = 4
        self.assertAlmostEqual(expr.grad[self.a], 1)
        self.assertAlmostEqual(expr.grad[b], 0)

        self.a.value = 3
        b.value = 0
        self.assertAlmostEqual(expr.grad[self.a], 0)
        self.assertAlmostEqual(expr.grad[b], 1)

        self.a.value = -1
        b.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 1)
        self.assertAlmostEqual(expr.grad[b], 0)

        y = Variable(2)
        expr = min_elemwise(self.x, y)
        self.x.value = [3, 4]
        y.value = [5, -5]
        val = np.zeros((2, 2)) + np.diag([1, 0])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)
        val = np.zeros((2, 2)) + np.diag([0, 1])
        self.assertItemsAlmostEqual(expr.grad[y].todense(), val)

        expr = min_elemwise(self.x, y)
        self.x.value = [-1e-9, 4]
        y.value = [1, 4]
        val = np.zeros((2, 2)) + np.diag([1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)
        val = np.zeros((2, 2)) + np.diag([0, 0])
        self.assertItemsAlmostEqual(expr.grad[y].todense(), val)

        expr = min_elemwise(self.A, self.B)
        self.A.value = [[1, 2], [3, 4]]
        self.B.value = [[5, 1], [3, 2.3]]
        div = (self.A.value/self.B.value).A.ravel(order='F')
        val = np.zeros((4, 4)) + np.diag([1, 0, 1, 0])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)
        val = np.zeros((4, 4)) + np.diag([0, 1, 0, 1])
        self.assertItemsAlmostEqual(expr.grad[self.B].todense(), val)

    def test_power(self):
        """Test domain for power.
        """
        expr = sqrt(self.a)
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 0.5/np.sqrt(2))

        self.a.value = 3
        self.assertAlmostEqual(expr.grad[self.a], 0.5/np.sqrt(3))

        self.a.value = -1
        self.assertAlmostEqual(expr.grad[self.a], None)

        expr = (self.x)**3
        self.x.value = [3, 4]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(),
                                    np.matrix("27 0; 0 48"))

        expr = (self.x)**3
        self.x.value = [-1e-9, 4]
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), np.matrix("0 0; 0 48"))

        expr = (self.A)**2
        self.A.value = [[1, -2], [3, 4]]
        val = np.zeros((4, 4)) + np.diag([2, -4, 6, 8])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

        # Constant.
        expr = (self.a)**0
        self.assertAlmostEqual(expr.grad[self.a], 0)

        expr = (self.x)**0
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), np.zeros((2, 2)))

    def test_partial_problem(self):
        """Test grad for partial minimization/maximization problems.
        """
        for obj in [Minimize((self.a)**-1), Maximize(entr(self.a))]:
            prob = Problem(obj, [self.x + self.a >= [5, 8]])
            # Optimize over nothing.
            expr = cvxpy.partial_optimize(prob, dont_opt_vars=[self.x, self.a])
            self.a.value = None
            self.x.value = None
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], None)
            self.assertAlmostEqual(grad[self.x], None)
            # Outside domain.
            self.a.value = 1.0
            self.x.value = [5, 5]
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], None)
            self.assertAlmostEqual(grad[self.x], None)

            self.a.value = 1
            self.x.value = [10, 10]
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], obj.args[0].grad[self.a])
            self.assertItemsAlmostEqual(grad[self.x].todense(), [0, 0, 0, 0])

            # Optimize over x.
            expr = cvxpy.partial_optimize(prob, opt_vars=[self.x])
            self.a.value = 1
            grad = expr.grad
            self.assertAlmostEqual(grad[self.a], obj.args[0].grad[self.a] + 0)

            # Optimize over a.
            fix_prob = Problem(obj, [self.x + self.a >= [5, 8], self.x == 0])
            fix_prob.solve()
            dual_val = fix_prob.constraints[0].dual_variable.value
            expr = cvxpy.partial_optimize(prob, opt_vars=[self.a])
            self.x.value = [0, 0]
            grad = expr.grad
            self.assertItemsAlmostEqual(grad[self.x].todense(), dual_val)

            # Optimize over x and a.
            expr = cvxpy.partial_optimize(prob, opt_vars=[self.x, self.a])
            grad = expr.grad
            self.assertAlmostEqual(grad, {})

    def test_affine(self):
        """Test grad for affine atoms.
        """
        expr = -self.a
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], -1)

        expr = 2*self.a
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 2)

        expr = self.a/2
        self.a.value = 2
        self.assertAlmostEqual(expr.grad[self.a], 0.5)

        expr = -(self.x)
        self.x.value = [3, 4]
        val = np.zeros((2, 2)) - np.diag([1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = -(self.A)
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 4)) - np.diag([1, 1, 1, 1])
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

        expr = self.A[0, 1]
        self.A.value = [[1, 2], [3, 4]]
        val = np.zeros((4, 1))
        val[2] = 1
        self.assertItemsAlmostEqual(expr.grad[self.A].todense(), val)

        z = Variable(3)
        expr = vstack(self.x, z)
        self.x.value = [1, 2]
        z.value = [1, 2, 3]
        val = np.zeros((2, 5))
        val[:, 0:2] = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        val = np.zeros((3, 5))
        val[:, 2:] = np.eye(3)
        self.assertItemsAlmostEqual(expr.grad[z].todense(), val)

        # cumsum
        expr = cumsum(self.x)
        self.x.value = [1, 2]
        val = np.ones((2, 2))
        val[1, 0] = 0
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)

        expr = cumsum(self.x, axis=1)
        self.x.value = [1, 2]
        val = np.eye(2)
        self.assertItemsAlmostEqual(expr.grad[self.x].todense(), val)
