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
class TestAffineAtoms:

    def test_reshape_C_order(self):
        """reshape with order='C' uses the transpose-based decomposition."""
        np.random.seed(0)
        m, n = 3, 4
        X = cp.Variable((m, n), bounds=[-1, 1], name='X')
        X.value = np.random.rand(m, n)
        obj = cp.sum(cp.reshape(cp.sin(X), (m * n,), order='C'))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_vstack(self):
        """vstack of nonlinear expressions."""
        np.random.seed(0)
        n = 5
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        stacked = cp.vstack([cp.sin(x), cp.cos(x)])
        obj = cp.sum(stacked)
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_upper_tri(self):
        """upper_tri of a nonlinear square matrix expression."""
        np.random.seed(0)
        n = 4
        X = cp.Variable((n, n), bounds=[-1, 1], name='X')
        X.value = np.random.rand(n, n)
        obj = cp.sum(cp.upper_tri(cp.sin(X)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_diag_mat(self):
        """diag(X) on a square matrix: matrix -> vector diagonal."""
        np.random.seed(0)
        n = 4
        X = cp.Variable((n, n), bounds=[-1, 1], name='X')
        X.value = np.random.rand(n, n)
        obj = cp.sum(cp.diag(cp.sin(X)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_div_by_scalar(self):
        """Scalar DivExpression: x / c with scalar c."""
        np.random.seed(0)
        n = 5
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        obj = cp.sum(cp.sin(x) / 3.0)
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_div_by_vector(self):
        """Elementwise DivExpression: x / c with vector c."""
        np.random.seed(0)
        n = 5
        c = np.arange(1, n + 1, dtype=float)
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        obj = cp.sum(cp.sin(x) / c)
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_conv_constant_kernel(self):
        """cp.conv(kernel, f(x)) with a constant 1D kernel."""
        np.random.seed(0)
        n = 6
        kernel = np.array([0.5, -1.0, 0.25])
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        obj = cp.sum(cp.conv(kernel, cp.sin(x)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_convolve_alias(self):
        """cp.convolve routes to the same converter as cp.conv."""
        np.random.seed(0)
        n = 6
        kernel = np.array([0.3, 0.6, -0.2])
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        obj = cp.sum(cp.convolve(kernel, cp.sin(x)))
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()

    def test_matmul_1d_dot_product(self):
        """x.T @ x for a 1D variable. Pins both the transpose 1D no-op and
        the var-var matmul handling of a 1D right operand (numpy column-vector
        convention)."""
        np.random.seed(0)
        n = 5
        x = cp.Variable(n, bounds=[-1, 1], name='x')
        x.value = np.random.rand(n)
        assert (cp.sin(x).T @ cp.sin(x)).shape == ()

        obj = cp.sin(x).T @ cp.sin(x)
        problem = cp.Problem(cp.Minimize(obj))

        problem.solve(solver=cp.IPOPT, nlp=True, verbose=False)
        assert(problem.status == cp.OPTIMAL)

        checker = DerivativeChecker(problem)
        checker.run_and_assert()
