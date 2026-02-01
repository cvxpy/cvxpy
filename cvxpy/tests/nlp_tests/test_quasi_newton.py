"""
Tests for quasi-Newton (L-BFGS) mode in NLP solvers.

These tests verify that the quasi-Newton (hessian_approximation='limited-memory')
mode works correctly with IPOPT. In this mode, the Hessian is approximated using
L-BFGS rather than computed exactly, which can be faster for large problems
and allows solving problems where the Hessian is difficult to compute.
"""

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestQuasiNewton:
    """Tests for quasi-Newton (L-BFGS) mode with IPOPT."""

    def test_rosenbrock_lbfgs(self):
        """Test Rosenbrock function with L-BFGS - classic unconstrained optimization."""
        x = cp.Variable(2, name='x')
        objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
        problem = cp.Problem(objective, [])
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-4)

    def test_hs071_lbfgs(self):
        """Test HS071 problem with L-BFGS - standard NLP test problem with constraints."""
        x = cp.Variable(4, bounds=[0, 6])
        x.value = np.array([1.0, 5.0, 5.0, 1.0])
        objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])

        constraints = [
            x[0]*x[1]*x[2]*x[3] >= 25,
            cp.sum(cp.square(x)) == 40,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        assert problem.status == cp.OPTIMAL
        # L-BFGS may converge to a slightly different point, use looser tolerance
        expected = np.array([0.75450865, 4.63936861, 3.78856881, 1.88513184])
        assert np.allclose(x.value, expected, atol=1e-3)

    def test_portfolio_lbfgs(self):
        """Test portfolio optimization with L-BFGS - quadratic objective."""
        r = np.array([0.026002150277777, 0.008101316405671, 0.073715909491990])
        Q = np.array([
            [0.018641039983891, 0.003598532927677, 0.001309759253660],
            [0.003598532927677, 0.006436938322676, 0.004887265158407],
            [0.001309759253660, 0.004887265158407, 0.068682765454814],
        ])
        x = cp.Variable(3)
        x.value = np.array([10.0, 10.0, 10.0])
        variance = cp.quad_form(x, Q)
        expected_return = r @ x
        problem = cp.Problem(
            cp.Minimize(variance),
            [
                cp.sum(x) <= 1000,
                expected_return >= 50,
                x >= 0
            ]
        )
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        assert problem.status == cp.OPTIMAL
        expected = np.array([4.97045504e+02, 0.0, 5.02954496e+02])
        assert np.allclose(x.value, expected, atol=1e-3)

    def test_analytic_center_lbfgs(self):
        """Test analytic center problem with L-BFGS - log-barrier problem."""
        np.random.seed(0)
        m, n = 50, 4
        b = np.ones(m)
        rand = np.random.randn(m - 2*n, n)
        A = np.vstack((rand, np.eye(n), np.eye(n) * -1))

        x = cp.Variable(n)
        objective = cp.Minimize(-cp.sum(cp.log(b - A @ x)))
        problem = cp.Problem(objective, [])
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        assert problem.status == cp.OPTIMAL

    def test_exact_vs_lbfgs_solution_quality(self):
        """Compare solution quality between exact Hessian and L-BFGS."""
        x = cp.Variable(2, name='x')
        objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)

        # Solve with exact Hessian
        problem_exact = cp.Problem(objective, [])
        problem_exact.solve(solver=cp.IPOPT, nlp=True,
                            hessian_approximation='exact')
        x_exact = x.value.copy()

        # Solve with L-BFGS
        problem_lbfgs = cp.Problem(objective, [])
        problem_lbfgs.solve(solver=cp.IPOPT, nlp=True,
                            hessian_approximation='limited-memory')
        x_lbfgs = x.value.copy()

        # Both should converge to the same solution
        assert np.allclose(x_exact, x_lbfgs, atol=1e-4)
        # Both should be close to [1, 1]
        assert np.allclose(x_exact, np.array([1.0, 1.0]), atol=1e-4)
        assert np.allclose(x_lbfgs, np.array([1.0, 1.0]), atol=1e-4)

    def test_large_scale_lbfgs(self):
        """Test L-BFGS on a larger problem where it's more useful."""
        np.random.seed(42)
        n = 100

        # Create a simple quadratic problem with better scaling
        Q = np.random.randn(n, n)
        Q = Q.T @ Q / n  # Make positive semidefinite and scale
        c = np.random.randn(n) / n

        x = cp.Variable(n)
        # Initialize to a feasible point
        x.value = np.ones(n) / n
        objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
        constraints = [cp.sum(x) == 1, x >= 0]
        problem = cp.Problem(objective, constraints)

        # Solve with L-BFGS
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory',
                      print_level=0)
        assert problem.status == cp.OPTIMAL

    def test_socp_lbfgs(self):
        """Test second-order cone problem with L-BFGS."""
        x = cp.Variable(3)
        y = cp.Variable()

        objective = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
        constraints = [
            cp.norm(x, 2) <= y,
            x[0] + x[1] + 3*x[2] >= 1.0,
            y <= 5
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        assert problem.status == cp.OPTIMAL
        assert np.allclose(objective.value, -13.548638814247532, atol=1e-3)

    def test_constrained_log_lbfgs(self):
        """Test constrained log problem with L-BFGS."""
        np.random.seed(123)
        n = 20

        x = cp.Variable(n, pos=True)
        x.value = np.ones(n) / n

        objective = cp.Minimize(-cp.sum(cp.log(x)))
        constraints = [cp.sum(x) == 1]
        problem = cp.Problem(objective, constraints)

        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        assert problem.status == cp.OPTIMAL
        # Optimal solution is uniform: x_i = 1/n
        assert np.allclose(x.value, np.ones(n) / n, atol=1e-4)

    def test_entropy_lbfgs(self):
        """Test entropy minimization with L-BFGS (nonconvex)."""
        np.random.seed(0)
        n = 10
        q = cp.Variable((n, ), nonneg=True)
        q.value = np.random.rand(n)
        q.value = q.value / np.sum(q.value)
        obj = cp.sum(cp.entr(q))
        constraints = [cp.sum(q) == 1]
        problem = cp.Problem(cp.Minimize(obj), constraints)
        problem.solve(solver=cp.IPOPT, nlp=True,
                      hessian_approximation='limited-memory')
        # Minimum entropy distribution is concentrated on one point
        assert np.sum(q.value > 1e-8) == 1
