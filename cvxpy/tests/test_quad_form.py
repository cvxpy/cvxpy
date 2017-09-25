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
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")

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
        self.assertEqual(str(cm.exception), "Problem does not follow DCP rules.")
