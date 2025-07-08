import numpy as np
import pandas as pd

import cvxpy as cp


class TestSmoothCanons():
    
    def test_max(self):
        x = cp.Variable(1)
        y = cp.Variable(1)

        objective = cp.Maximize(cp.maximum(x, y))

        constraints = [x - 14 == 0, y - 6 == 0]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert problem.value == 14
        assert x.value == 14
        assert y.value == 6

    def test_min(self):
        x = cp.Variable(1)
        y = cp.Variable(1)

        objective = cp.Minimize(cp.minimum(x, y))

        constraints = [x - 14 == 0, y - 6 == 0]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert problem.value == 6


class TestExamplesIPOPT():
    
    def test_hs071(self):
        x = cp.Variable(4, bounds=[0,6])
        x.value = np.array([1.0, 5.0, 5.0, 1.0])
        objective = cp.Minimize(x[0]*x[3]*(x[0] + x[1] + x[2]) + x[2])
        
        # Constraints
        constraints = [
            x[0]*x[1]*x[2]*x[3] >= 25,  # Product constraint
            cp.sum_squares(x) == 40,    # Sum of squares constraint
        ]
        # Create problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([0.75450865, 4.63936861, 3.78856881, 1.88513184]))

    def test_portfolio_opt(self):
        df = pd.DataFrame({
        'IBM': [93.043, 84.585, 111.453, 99.525, 95.819, 114.708, 111.515,
                113.211, 104.942, 99.827, 91.607, 107.937, 115.590],
        'WMT': [51.826, 52.823, 56.477, 49.805, 50.287, 51.521, 51.531,
                48.664, 55.744, 47.916, 49.438, 51.336, 55.081],
        'SEHI': [1.063, 0.938, 1.000, 0.938, 1.438, 1.700, 2.540, 2.390,
                3.120, 2.980, 1.900, 1.750, 1.800]
        })

        # Compute returns
        returns = df.pct_change().dropna().values
        r = np.mean(returns, axis=0)
        Q = np.cov(returns.T)

        # Single-objective optimization
        x = cp.Variable(3)  # Non-negative weights
        x.value = np.array([10.0, 10.0, 10.0])  # Initial guess
        variance = cp.quad_form(x, Q)
        expected_return = r @ x

        problem = cp.Problem(
        cp.Minimize(variance),[cp.sum(x) <= 1000, expected_return >= 50, x >= 0])
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([4.97045504e+02, -9.89291685e-09, 5.02954496e+02]))

    def test_mle(self):
        n = 1000
        np.random.seed(1234)
        data = np.random.randn(n)
        
        mu = cp.Variable(1, name="mu")
        mu.value = np.array([0.0])
        sigma = cp.Variable(1, name="sigma")
        sigma.value = np.array([1.0])

        constraints = [mu == sigma**2, sigma >= 1e-6]
        # Sum of squared residuals
        residual_sum = cp.sum_squares(data - mu)
        log_likelihood = (
            (n / 2) * cp.log(1 / (2 * np.pi * (sigma)**2))
            - residual_sum / (2 * (sigma)**2)
        )
        
        objective = cp.Minimize(-log_likelihood)
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(mu.value, 0.77079388, atol=1e-5)
        assert np.allclose(sigma.value, 0.59412321, atol=1e-5)

    def test_rosenbrock(self):
        x = cp.Variable(2)
        objective = cp.Minimize((1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2)
        problem = cp.Problem(objective, [])
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(x.value, np.array([1.0, 1.0]), atol=1e-5)


class TestNonlinearControl():
    pass

