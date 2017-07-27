"""
Copyright 2016 Jaehyun Park

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

from cvxpy import *
from cvxpy.tests.base_test import BaseTest
import unittest
import numpy as np
import scipy.sparse as sp


class TestLS(BaseTest):
    """ LS solver tests. """

    def test_regression(self):
        # Set the random seed to get consistent data
        np.random.seed(1)

        # Number of examples to use
        n = 100

        # Specify the true value of the variable
        true_coeffs = np.matrix('2; -2; 0.5')

        # Generate data
        x_data = np.random.rand(n, 1) * 5
        x_data = np.asmatrix(x_data)
        x_data_expanded = np.hstack([np.power(x_data, i) for i in range(1, 4)])
        x_data_expanded = np.asmatrix(x_data_expanded)
        y_data = x_data_expanded * true_coeffs + 0.5 * np.random.rand(n, 1)
        y_data = np.asmatrix(y_data)

        slope = Variable()
        offset = Variable()
        line = offset + x_data * slope
        residuals = line - y_data
        fit_error = sum_squares(residuals)
        optval = Problem(Minimize(fit_error), []).solve(solver=LS)
        self.assertAlmostEqual(optval, 1171.60037715)

        quadratic_coeff = Variable()
        slope = Variable()
        offset = Variable()
        quadratic = offset + x_data * slope + quadratic_coeff * np.power(x_data, 2)
        residuals = quadratic - y_data
        fit_error = sum_squares(residuals)
        optval = Problem(Minimize(fit_error), []).solve(solver=LS)
        optval2 = Problem(Minimize(fit_error), []).solve(solver=ECOS)
        self.assertAlmostEqual(optval, 139.225660756)

    def test_control(self):
        # Some constraints on our motion
        # The object should start from the origin, and end at rest
        initial_velocity = np.matrix('-20; 100')
        final_position = np.matrix('100; 100')

        T = 100  # The number of timesteps
        h = 0.1  # The time between time intervals
        mass = 1  # Mass of object
        drag = 0.1  # Drag on object
        g = np.matrix('0; -9.8')  # Gravity on object

        # Declare the variables we need
        position = Variable(2, T)
        velocity = Variable(2, T)
        force = Variable(2, T - 1)

        # Create a problem instance
        mu = 1
        constraints = []

        # Add constraints on our variables
        for i in range(T - 1):
            constraints.append(position[:, i + 1] == position[:, i] + h * velocity[:, i])
            acceleration = force[:, i]/mass + g - drag * velocity[:, i]
            constraints.append(velocity[:, i + 1] == velocity[:, i] + h * acceleration)

        # Add position constraints
        constraints.append(position[:, 0] == 0)
        constraints.append(position[:, -1] == final_position)

        # Add velocity constraints
        constraints.append(velocity[:, 0] == initial_velocity)
        constraints.append(velocity[:, -1] == 0)

        # Solve the problem
        optval = Problem(Minimize(sum_squares(force)), constraints).solve(solver=LS)
        self.assertAlmostEqual(optval, 17850.0, places=0)

    def test_sparse_system(self):
        m = 1000
        n = 800
        r = 700
        np.random.seed(1)
        density = 0.2
        A = sp.rand(m, n, density)
        b = np.random.randn(m, 1)
        G = sp.rand(r, n, density)
        h = np.random.randn(r, 1)

        x = Variable(n)
        optval = Problem(Minimize(sum_squares(A*x - b)), [G*x == h]).solve(solver=LS)
        self.assertAlmostEqual(optval, 6071.830658)

    def test_equivalent_forms(self):
        m = 100
        n = 80
        r = 70
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m, 1)
        G = np.random.randn(r, n)
        h = np.random.randn(r, 1)

        # ||Ax-b||^2 = x^T (A^T A) x - 2(A^T b)^T x + ||b||^2
        P = np.dot(A.T, A)
        q = -2*np.dot(A.T, b)
        r = np.dot(b.T, b)

        Pinv = np.linalg.inv(P)

        x = Variable(n)

        obj1 = sum_squares(A*x - b)
        obj2 = sum_entries(square(A*x - b))
        obj3 = quad_form(x, P)+q.T*x+r
        obj4 = matrix_frac(x, Pinv)+q.T*x+r

        cons = [G*x == h]

        v1 = Problem(Minimize(obj1), cons).solve(solver=LS)
        v2 = Problem(Minimize(obj2), cons).solve(solver=LS)
        v3 = Problem(Minimize(obj3), cons).solve(solver=LS)
        v4 = Problem(Minimize(obj4), cons).solve(solver=LS)
        self.assertAlmostEqual(v1, 681.119420108)
        self.assertAlmostEqual(v2, 681.119420108)
        self.assertAlmostEqual(v3, 681.119420108)
        self.assertAlmostEqual(v4, 681.119420108)

    def test_smooth_ridge(self):
        np.random.seed(1)
        n = 500
        k = 50
        delta = 1
        eta = 1

        A = np.random.rand(k, n)
        b = np.random.rand(k, 1)
        x = Variable(n)
        obj = sum_squares(A*x - b) + delta*sum_squares(x[:-1]-x[1:]) + eta*sum_squares(x)
        optval = Problem(Minimize(obj), []).solve(solver=LS)
        self.assertAlmostEqual(optval, 0.24989717371)
