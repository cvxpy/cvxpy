"""
Copyright, the CVXPY authors

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


import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestCompositions():
    # Stress tests for affine matrix atoms in the diff engine.
    
    def test_left_matmul_composition(self):
        np.random.seed(0)
        X = cp.Variable((10, 10), bounds = [-0.2, 0.2])
        A = np.random.rand(10, 10)
        obj = cp.Minimize(cp.Trace(cp.exp(A @ X)))
        constraints = [X[1, 1] + X[2, 2] == 0.1]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_right_matmul_composition(self):
        np.random.seed(0)
        X = cp.Variable((10, 10), bounds = [-0.2, 0.2])
        A = np.random.rand(10, 10)
        obj = cp.Minimize(cp.sum(cp.exp(X @ A)))
        constraints = [X[1, 1] + X[2, 2] == 0.1]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_multiply_linear_composition(self):
        m = 20
        n = 5
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds = [-1, 1])
        Y = cp.Variable((n, n), bounds = [-1, 1])
        X.value = np.random.rand(n, n)
        Y.value = np.random.rand(n, n)
        obj = cp.Minimize(cp.sum(cp.multiply(A @ X, B @ Y)))
        prob = cp.Problem(obj)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    def test_multiply_nonlinear_composition(self):
        m = 20
        n = 5
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds = [-1, 1])
        Y = cp.Variable((n, n), bounds = [-1, 1])
        X.value = np.random.rand(n, n)
        Y.value = np.random.rand(n, n)
        obj = cp.Minimize(cp.sum(cp.multiply(cp.square(A @ X), cp.logistic(B @ Y))))
        prob = cp.Problem(obj)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    def test_multiply_nonlinear_composition_transpose(self):
        m = 10
        n = 10
        A = np.random.rand(m, n)
        B = np.random.rand(m, n)
        X = cp.Variable((n, n), bounds = [-1, 1])
        Y = cp.Variable((n, n), bounds = [-1, 1])
        X.value = np.random.rand(n, n)
        Y.value = np.random.rand(n, n)
        obj = cp.Minimize(cp.sum(cp.multiply(cp.square((A @ X).T), cp.logistic(B @ Y))))
        prob = cp.Problem(obj)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
                    
    def test_quad_form_composition(self):
        n = 25
        Q = np.random.rand(n, n)
        Q = Q + Q.T
        x = cp.Variable(n, bounds=[-1, 1])
        obj = cp.Minimize(cp.quad_form(cp.multiply(cp.sin(x), x), Q))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
    
    def test_quad_form_composition_two(self):
        n = 10
        Q = np.random.rand(n, n)
        Q = Q + Q.T
        x = cp.Variable(n, bounds=[-1, 1])
        y = cp.Variable(n, bounds=[-1, 1])
        obj = cp.Minimize(cp.quad_form(cp.multiply(cp.sin(x), cp.multiply(x, y)), Q))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n)
        y.value = np.random.rand(n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_matmul_composition_one(self):
        np.random.seed(0)
        m, n, p = 5, 7, 11
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        Y.value = np.random.rand(n, p)
        obj = cp.sum(cp.matmul(X, cp.cos(Y)))
        problem = cp.Problem(cp.Minimize(obj))
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        assert(problem.status == cp.OPTIMAL)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_matmul_composition_two(self):
        np.random.seed(0)
        m, n, p = 5, 5, 5
        X = cp.Variable((m, n), bounds=[-1, 1])
        Y = cp.Variable((n, p), bounds=[-2, 2])
        Y.value = np.random.rand(n, p)
        X.value = np.random.rand(m, n)
        obj = cp.sum(cp.matmul(cp.matmul(X, X), cp.cos(Y) + X))
        problem = cp.Problem(cp.Minimize(obj))
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        assert(problem.status == cp.OPTIMAL)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
    
    def test_matmul_composition_three(self):
        np.random.seed(0)
        m, n, p = 5, 5, 5
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        Y = cp.Variable((n, p), bounds=[-2, 2], name='Y')
        Y.value = np.random.rand(n, p)
        X.value = np.random.rand(m, n)
        obj = cp.sum(cp.matmul(cp.matmul(X, X.T), (cp.cos(Y) + X).T))
        problem = cp.Problem(cp.Minimize(obj))
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        assert(problem.status == cp.OPTIMAL)
        checker = DerivativeChecker(problem)
        checker.run_and_assert()
    
    def test_div(self):
        n = 5
        x = cp.Variable((n, 1), bounds=[1, 2])
        A = np.random.rand(n, n)
        obj = cp.Minimize(cp.sum(1 / (x / A @ x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, 1) + 1
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=True)
        assert np.allclose(x.value, 2 * np.ones((n, 1)))
