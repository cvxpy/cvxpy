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

from __future__ import division, print_function, absolute_import

import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.sparse as sp
import cvxpy
import warnings

from cvxpy.tests.base_test import BaseTest


class TestNonOptimal(BaseTest):
    def test_singular_quad_form(self):
        """Test quad form with a singular matrix.
        """
        # Solve a quadratic program.
        np.random.seed(1234)
        for n in (3, 4, 5):
            for i in range(5):

                # construct a random 1d finite distribution
                v = np.exp(np.random.randn(n))
                v = v / np.sum(v)

                # construct a random positive definite matrix
                A = np.random.randn(n, n)
                Q = np.dot(A, A.T)

                # Project onto the orthogonal complement of v.
                # This turns Q into a singular matrix with a known nullspace.
                E = np.identity(n) - np.outer(v, v) / np.inner(v, v)
                Q = np.dot(E, np.dot(Q, E.T))
                observed_rank = np.linalg.matrix_rank(Q)
                desired_rank = n-1
                yield assert_equal, observed_rank, desired_rank

                for action in 'minimize', 'maximize':

                    # Look for the extremum of the quadratic form
                    # under the simplex constraint.
                    x = cvxpy.Variable(n)
                    if action == 'minimize':
                        q = cvxpy.quad_form(x, Q)
                        objective = cvxpy.Minimize(q)
                    elif action == 'maximize':
                        q = cvxpy.quad_form(x, -Q)
                        objective = cvxpy.Maximize(q)
                    constraints = [0 <= x, cvxpy.sum(x) == 1]
                    p = cvxpy.Problem(objective, constraints)
                    p.solve()

                    # check that cvxpy found the right answer
                    xopt = x.value.A.flatten()
                    yopt = np.dot(xopt, np.dot(Q, xopt))
                    assert_allclose(yopt, 0, atol=1e-3)
                    assert_allclose(xopt, v, atol=1e-3)

    def test_sparse_quad_form(self):
        """Test quad form with a sparse matrix.
        """
        Q = sp.eye(2)
        x = cvxpy.Variable(2)
        cost = cvxpy.quad_form(x, Q)
        prob = cvxpy.Problem(cvxpy.Minimize(cost), [x == [1, 2]])
        self.assertAlmostEqual(prob.solve(), 5)

        # Here are our QP factors
        A = cvxpy.Constant(sp.eye(4))
        c = np.ones(4).reshape((1, 4))

        # Here is our optimization variable
        x = cvxpy.Variable(4)

        # And the QP problem setup
        function = cvxpy.quad_form(x, A) - cvxpy.matmul(c, x)
        objective = cvxpy.Minimize(function)
        problem = cvxpy.Problem(objective)

        problem.solve()
        self.assertEqual(len(function.value), 1)

    def test_param_quad_form(self):
        """Test quad form with a parameter.
        """
        P = cvxpy.Parameter((2, 2), PSD=True)
        Q = np.eye(2)
        x = cvxpy.Variable(2)
        cost = cvxpy.quad_form(x, P)
        P.value = Q
        prob = cvxpy.Problem(cvxpy.Minimize(cost), [x == [1, 2]])
        self.assertAlmostEqual(prob.solve(), 5)

    def test_non_symmetric(self):
        """Test when P is constant and not symmetric.
        """
        P = np.array([[2, 2], [3, 4]])
        x = cvxpy.Variable(2)
        cost = cvxpy.quad_form(x, P)
        prob = cvxpy.Problem(cvxpy.Minimize(cost), [x == [1, 2]])
        with self.assertRaises(Exception) as cm:
            prob.solve()
        self.assertTrue("Problem does not follow DCP rules."
                        in str(cm.exception))

    def test_non_psd(self):
        """Test error when P is symmetric but not definite.
        """
        P = np.array([[1, 0], [0, -1]])
        x = cvxpy.Variable(2)
        # Forming quad_form is okay
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cost = cvxpy.quad_form(x, P)
        prob = cvxpy.Problem(cvxpy.Minimize(cost), [x == [1, 2]])
        with self.assertRaises(Exception) as cm:
            prob.solve()
        self.assertTrue("Problem does not follow DCP rules."
                        in str(cm.exception))

    def test_obj_eval(self):
        """Test case where objective evaluation differs from result.
        """
        x = cvxpy.Variable((2, 1))
        A = np.array([[1.0]])
        B = np.array([[1.0, 1.0]]).T
        obj0 = -B.T * x
        obj1 = cvxpy.quad_form(B.T * x, A)
        prob = cvxpy.Problem(cvxpy.Minimize(obj0 + obj1))
        prob.solve()
        self.assertAlmostEqual(prob.value, prob.objective.value)

    def test_zero_term(self):
        """Test a quad form multiplied by zero.
        """
        data_norm = np.random.random(5)
        M = np.random.random(5*2).reshape((5, 2))
        c = cvxpy.Variable(M.shape[1])
        lopt = 0
        laplacian_matrix = np.ones((2, 2))
        design_matrix = cvxpy.Constant(M)
        objective = cvxpy.Minimize(
            cvxpy.sum_squares(design_matrix * c - data_norm) +
            lopt * cvxpy.quad_form(c, laplacian_matrix)
        )
        constraints = [(M[0] * c) == 1]  # (K * c) >= -0.1]
        prob = cvxpy.Problem(objective, constraints)
        prob.solve()
