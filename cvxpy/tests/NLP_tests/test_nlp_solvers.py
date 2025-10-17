
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestExamplesIPOPT:
    """
    Nonlinear test problems taken from the IPOPT documentation and
    the Julia documentation: https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/simple_examples/.
    """
    def test_hs071(self):
        x = cp.Variable(4, bounds=[0,6])
        x.value = np.array([1.0, 5.0, 5.0, 1.0])
        objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])

        constraints = [
            x[0]*x[1]*x[2]*x[3] >= 25,
            cp.sum(cp.square(x)) == 40,
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([0.75450865, 4.63936861, 3.78856881, 1.88513184]))

    def test_mle(self):
        n = 1000
        np.random.seed(1234)
        data = np.random.randn(n)
        
        mu = cp.Variable((1, ), name="mu")
        mu.value = np.array([0.0])
        sigma = cp.Variable((1, ), name="sigma")
        sigma.value = np.array([1.0])

        constraints = [mu == sigma**2]
        #residual_sum = cp.sum_squares(data - mu)
        log_likelihood = (
            (n / 2) * cp.log(1 / (2 * np.pi * (sigma)**2))
            - cp.sum(cp.square(data-mu)) / (2 * (sigma)**2)
        )
        
        objective = cp.Maximize(log_likelihood)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(sigma.value, 0.77079388)
        assert np.allclose(mu.value, 0.59412321)

    def test_portfolio_opt(self):
        # data taken from https://jump.dev/JuMP.jl/stable/tutorials/nonlinear/portfolio/
        # r and Q are pre-computed from historical data of 3 assets
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
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([4.97045504e+02, -9.89291685e-09, 5.02954496e+02]))

    def test_rosenbrock(self):
        x = cp.Variable(2, name='x')
        objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
        problem = cp.Problem(objective, [])
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]))

    def test_qcp(self):
        x = cp.Variable(1)
        y = cp.Variable(1, bounds=[0, np.inf])
        z = cp.Variable(1, bounds=[0, np.inf])

        objective = cp.Maximize(x)
        
        constraints = [
            x + y + z == 1,
            x**2 + y**2 - z**2 <= 0,
            x**2 - cp.multiply(y, z) <= 0
        ]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([0.32699284]))
        assert np.allclose(y.value, np.array([0.25706586]))
        assert np.allclose(z.value, np.array([0.4159413]))

    def test_analytic_polytope_center(self):
        # Generate random data
        np.random.seed(0)
        m, n = 50, 4
        b = np.ones(m)
        rand = np.random.randn(m - 2*n, n)
        A = np.vstack((rand, np.eye(n), np.eye(n) * -1))
        """
        m, n = 5, 2
        A = np.array([[1, 0], [-1, 0], [0, 1], [0, -1], [-0.5, 1]])
        b = np.array([1, 1, 1, 1, 0.5])
        """
        # Define the variable
        x = cp.Variable(n)
        # set initial value for x
        objective = cp.Minimize(-cp.sum(cp.log(b - A @ x)))
        problem = cp.Problem(objective, [])
        # Solve the problem
        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact')
        assert problem.status == cp.OPTIMAL

    @pytest.mark.xfail(reason="Fails because norm is not supported yet.")
    def test_socp(self):
        # Define variables
        x = cp.Variable(3)
        y = cp.Variable()

        # Define objective function
        objective = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])

        # Define constraints
        constraints = [
            cp.norm(x, 2) <= y,
            x[0] + x[1] + 3*x[2] >= 1.0,
            y <= 5
        ]

        # Create and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)

class TestNonlinearControl:
    
    def test_clnlbeam(self):
        N = 1000
        h = 1 / N
        alpha = 350

        t = cp.Variable(N+1, bounds=[-1, 1])
        x = cp.Variable(N+1, bounds=[-0.05, 0.05])
        u = cp.Variable(N+1)

        control_terms = cp.multiply(0.5 * h, cp.power(u[1:], 2) + cp.power(u[:-1], 2))
        trigonometric_terms = cp.multiply(0.5 * alpha * h, cp.cos(t[1:]) + cp.cos(t[:-1]))
        objective_terms = cp.sum(control_terms + trigonometric_terms)

        objective = cp.Minimize(objective_terms)
        constraints = []
        position_constraints = (x[1:] - x[:-1] - 
                            cp.multiply(0.5 * h, cp.sin(t[1:]) + cp.sin(t[:-1])) == 0)
        constraints.append(position_constraints)
        angle_constraint = (t[1:] - t[:-1] - 0.5 * h * (u[1:] + u[:-1]) == 0)
        constraints.append(angle_constraint)

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True,
                    derivative_test='none')
        assert problem.status == cp.OPTIMAL
        assert np.allclose(problem.value, 3.500e+02)
