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

from cvxpy import *
import numpy as np
import scipy.sparse as sp
import scipy.linalg as LA
import unittest
from base_test import BaseTest

class TestMatFree(BaseTest):
    """ Unit tests for matrix-free solvers. """

    def test_basic(self):
        """Test SCS mat free solver.
        """
        n = 100
        x = Variable(n)
        np.random.seed(1)
        A = np.random.randn(2*n, n)
        b = A.dot(np.ones((n, 1)))
        fit = norm(A*x - b)
        prob = Problem(Minimize(fit), [])
        result = prob.solve(solver=SCS_MAT_FREE, equil_steps=10,
                            verbose=True)
        fit1 = fit.value
        result2 = prob.solve(solver=CVXOPT)
        fit2 = fit.value
        self.assertAlmostEqual(fit1, fit2, places=1)

    def test_convolution(self):
        """Test matrix free convolution.
        """
        import random

        from math import pi, sqrt, exp

        def gauss(n=11,sigma=1):
            r = range(-int(n/2),int(n/2)+1)
            return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]

        np.random.seed(5)
        random.seed(5)
        DENSITY = 0.008
        n = 100
        x = Variable(n)
        # Create sparse signal.
        signal = np.zeros(n)
        nnz = 0
        for i in range(n):
            if random.random() < DENSITY:
                signal[i] = random.uniform(0, 100)
                nnz += 1

        # Gaussian kernel.
        m = 11
        kernel = gauss(m, m/10)

        # Noisy signal.
        std = 1
        noise = np.random.normal(scale=std, size=n+m-1)
        noisy_signal = conv(kernel, signal) + noise

        gamma = Parameter(sign="positive")
        fit = norm(conv(kernel, x) - noisy_signal, 2)
        regularization = norm(x, 1)
        constraints = [x >= 0]
        gamma.value = 0.06
        prob = Problem(Minimize(fit), constraints)
        result1 = prob.solve(solver=SCS_MAT_FREE, verbose=True,
            max_iters=2500, equil_steps=10, eps=1e-3)
        result2 = prob.solve(solver=CVXOPT, verbose=True)
        self.assertAlmostEqual(result1, result2, places=2)

    def test_mat_ineq(self):
        """Test matrix inequality problem.
        """
        m = 10
        n = 10
        k = 10
        A = np.random.randn(m, n)
        X = Variable(n, k)

        cost = sum_squares(X - 1)
        prob = Problem(Minimize(cost), [A*X >= 2])
        result1 = prob.solve(solver=SCS_MAT_FREE, verbose=True)
        result2 = prob.solve(solver=ECOS, verbose=True)
        self.assertAlmostEqual(result1, result2, places=1)
