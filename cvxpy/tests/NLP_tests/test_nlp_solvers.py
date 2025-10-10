import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestSmoothCanons:
    
    def test_max(self):
        x = cp.Variable(1)
        y = cp.Variable(1)

        objective = cp.Maximize(cp.maximum(x, y))

        constraints = [x - 14 == 0, y - 6 == 0]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert np.isclose(problem.value, 14)
        assert np.isclose(x.value, 14)
        assert np.isclose(y.value, 6)

    def test_min(self):
        x = cp.Variable(1)
        y = cp.Variable(1)

        objective = cp.Minimize(cp.minimum(x, y))

        constraints = [x - 14 == 0, y - 6 == 0]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.OPTIMAL
        assert problem.value == 6

    def test_max_2(self):
        # Define variables
        x = cp.Variable(3)
        y = cp.Variable(3)

        objective = cp.Maximize(cp.sum(cp.maximum(x, y)))

        constraints = [x <= 14, y <= 14]

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
        assert problem.status == cp.OPTIMAL
        assert np.allclose(y.value, 14)
        # need to set low tolerance, actual value is 42.00062
        assert np.allclose(problem.value, 42, atol=1e-3)


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

    """
    def test_portfolio_opt(self):
        df = pd.DataFrame({
        'IBM': [93.043, 84.585, 111.453, 99.525, 95.819, 114.708, 111.515,
                113.211, 104.942, 99.827, 91.607, 107.937, 115.590],
        'WMT': [51.826, 52.823, 56.477, 49.805, 50.287, 51.521, 51.531,
                48.664, 55.744, 47.916, 49.438, 51.336, 55.081],
        'SEHI': [1.063, 0.938, 1.000, 0.938, 1.438, 1.700, 2.540, 2.390,
                3.120, 2.980, 1.900, 1.750, 1.800]
        })

        returns = df.pct_change().dropna().values
        r = np.mean(returns, axis=0)
        Q = np.cov(returns.T)

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
    """

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

    """
    def test_acopf(self):
        N = 4

        # Conductance/susceptance components
        G = np.array(
            [
                [1.7647, -0.5882, 0.0, -1.1765],
                [-0.5882, 1.5611, -0.3846, -0.5882],
                [0.0, -0.3846, 1.5611, -1.1765],
                [-1.1765, -0.5882, -1.1765, 2.9412],
            ]
        )

        B = np.array(
            [
                [-7.0588, 2.3529, 0.0, 4.7059],
                [2.3529, -6.629, 1.9231, 2.3529],
                [0.0, 1.9231, -6.629, 4.7059],
                [4.7059, 2.3529, 4.7059, -11.7647],
            ]
        )

        # Assign bounds where fixings are needed
        v_lb = np.array([1.0, 0.0, 1.0, 0.0])
        v_ub = np.array([1.0, 1.5, 1.0, 1.5])

        P_lb = np.array([-3.0, -0.3, 0.3, -0.2])
        P_ub = np.array([3.0, -0.3, 0.3, -0.2])

        Q_lb = np.array([-3.0, -0.2, -3.0, -0.15])
        Q_ub = np.array([3.0, -0.2, 3.0, -0.15])

        theta_lb = np.array([0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2])
        theta_ub = np.array([0.0, np.pi / 2, np.pi / 2, np.pi / 2])

        # Create variables with bounds
        P = cp.Variable(N, name="P")  # Real power for buses
        Q = cp.Variable(N, name="Q")  # Reactive power for buses
        v = cp.Variable(N, name="v")  # Voltage magnitude at buses
        theta = cp.Variable(N, name="theta")  # Voltage angle at buses

        # Reshape theta to column vector for broadcasting
        theta_col = cp.reshape(theta, (N, 1))

        # Create constraints list
        constraints = []

        # Add bound constraints
        constraints += [
            P >= P_lb,
            P <= P_ub,
            Q >= Q_lb,
            Q <= Q_ub,
            v >= v_lb,
            v <= v_ub,
            theta >= theta_lb,
            theta <= theta_ub
        ]
        P_balance = cp.multiply(
            v,
            (
                G * cp.cos(theta_col - theta_col.T)
                + B * cp.sin(theta_col - theta_col.T)
            ) @ v
        )
        constraints.append(P == P_balance)

        # Reactive power balance
        Q_balance = cp.multiply(
            v,
            (
                G * cp.sin(theta_col - theta_col.T)
                - B * cp.cos(theta_col - theta_col.T)
            ) @ v
        )
        constraints.append(Q == Q_balance)

        # Objective: minimize reactive power at buses 1 and 3 (indices 0 and 2)
        objective = cp.Minimize(Q[0] + Q[2])

        # Create and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True)
        assert problem.status == cp.INFEASIBLE
    """

@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestNonlinearControl:
    
    def test_control_of_car(self):
        pass

    def test_clnlbeam(self):
        N = 10
        h = 1 / N
        alpha = 350
        
        t = cp.Variable(N+1)
        x = cp.Variable(N+1)
        u = cp.Variable(N+1)

        objective_terms = []
        for i in range(N):
            control_term = 0.5 * h * (u[i+1]**2 + u[i]**2)
            trigonometric_term = 0.5 * alpha * h * (cp.cos(t[i+1]) + cp.cos(t[i]))
            objective_terms.append(control_term + trigonometric_term)
        
        objective = cp.Minimize(cp.sum(objective_terms))
        
        constraints = [
            t >= -1,
            t <= 1,
            x >= -0.05,
            x <= 0.05
        ]
        
        for i in range(N):
            position_constraint = (x[i+1] - x[i] - 
                                0.5 * h * (cp.sin(t[i+1]) + cp.sin(t[i])) == 0)
            constraints.append(position_constraint)
            
            angle_constraint = (t[i+1] - t[i] - 
                            0.5 * h * u[i+1] - 0.5 * h * u[i] == 0)
            constraints.append(angle_constraint)
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.IPOPT, nlp=True, hessian_approximation='exact')
        assert problem.status == cp.OPTIMAL
        assert problem.value == 3.500e+02
