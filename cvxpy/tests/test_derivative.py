import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest

import numpy as np


def perturbcheck(problem, delta=1e-5, atol=1e-8, eps=1e-10):
    """Checks the analytical derivative against a numerical computation."""
    np.random.seed(0)
    # Compute perturbations analytically
    for param in problem.parameters():
        param.delta = delta * np.random.randn(*param.shape)
    problem.solve(requires_grad=True, eps=eps)
    problem.derivative()
    variable_values = [v.value for v in problem.variables()]
    deltas = [v.delta for v in problem.variables()]

    # Compute perturbations numerically
    old_values = {}
    for param in problem.parameters():
        old_values[param] = param.value
        param.value += param.delta
    problem.solve(cp.SCS, eps=eps)
    num_deltas = [
        v.value - old_value for (v, old_value)
        in zip(problem.variables(), variable_values)]

    for analytical, numerical in zip(deltas, num_deltas):
        np.testing.assert_allclose(analytical, numerical, atol=atol)

    for param in problem.parameters():
        param.value = old_values[param]


def gradcheck(problem, delta=1e-5, atol=1e-5, eps=1e-10):
    """Checks the analytical adjoint derivative against a numerical computation."""
    size = sum(p.size for p in problem.parameters())
    values = np.zeros(size)
    offset = 0
    for param in problem.parameters():
        values[offset:offset + param.size] = np.asarray(param.value).flatten()
        param.value = values[offset:offset + param.size].reshape(param.shape)
        offset += param.size

    numgrad = np.zeros(values.shape)
    for i in range(values.size):
        old = values[i]
        values[i] = old + 0.5 * delta
        problem.solve(cp.SCS, eps=eps)
        left_solns = [x.value for x in problem.variables()]

        values[i] = old - 0.5 * delta
        problem.solve(cp.SCS, eps=eps)
        right_solns = [x.value for x in problem.variables()]

        numgrad[i] = (np.sum(left_solns) - np.sum(right_solns)) / delta
        values[i] = old
    numgrads = []
    offset = 0
    for param in problem.parameters():
        numgrads.append(
            numgrad[offset:offset + param.size].reshape(param.shape))
        offset += param.size

    old_gradients = {}
    for x in problem.variables():
        old_gradients[x] = x.gradient
        x.gradient = None
    problem.solve(requires_grad=True, eps=eps)
    problem.backward()

    for param, numgrad in zip(problem.parameters(), numgrads):
        np.testing.assert_allclose(param.gradient, numgrad, atol=atol)

    for x in problem.variables():
        x.gradient = old_gradients[x]


class TestBackward(BaseTest):
    """Test problem.backward() and problem.derivative()."""
    def setUp(self):
        try:
            import diffcp
            diffcp  # for flake8
        except ImportError:
            self.skipTest("diffcp not installed.")

    def test_scalar_quadratic(self):
        b = cp.Parameter()
        x = cp.Variable()
        quadratic = cp.square(x - 2 * b)
        problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])
        b.value = 3.
        problem.solve(requires_grad=True, eps=1e-10)
        self.assertAlmostEqual(x.value, 6.)
        problem.backward()

        # x* = 2 * b, dx*/db = 2
        # x.gradient == None defaults to 1.0
        self.assertAlmostEqual(b.gradient, 2.)
        x.gradient = 4.
        problem.backward()
        self.assertAlmostEqual(b.gradient, 8.)
        gradcheck(problem, atol=1e-4)
        perturbcheck(problem, atol=1e-4)

        problem.solve(requires_grad=True, eps=1e-10)
        b.delta = 1e-3
        problem.derivative()
        self.assertAlmostEqual(x.delta, 2e-3)

    def test_l1_square(self):
        np.random.seed(0)
        n = 3
        x = cp.Variable(n)
        A = cp.Parameter((n, n))
        b = cp.Parameter(n, name='b')
        objective = cp.Minimize(cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective)
        self.assertTrue(problem.is_dpp())

        L = np.random.randn(n, n)
        A.value = L.T @ L + np.eye(n)
        b.value = np.random.randn(n)
        gradcheck(problem)
        perturbcheck(problem)

    def test_l1_rectangle(self):
        np.random.seed(0)
        m, n = 3, 2
        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m, name='b')
        objective = cp.Minimize(cp.pnorm(A @ x - b, p=1))
        problem = cp.Problem(objective)
        self.assertTrue(problem.is_dpp())

        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        gradcheck(problem)
        perturbcheck(problem)

    def test_least_squares(self):
        np.random.seed(0)
        m, n = 20, 5
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(obj))

        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        gradcheck(problem)
        perturbcheck(problem)

    def test_logistic_regression(self):
        np.random.seed(0)
        N, n = 5, 2
        X_np = np.random.randn(N, n)
        a_true = np.random.randn(n, 1)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        y = np.round(sigmoid(X_np @ a_true + np.random.randn(N, 1) * 0.5))

        a = cp.Variable((n, 1))
        X = cp.Parameter((N, n))
        lam = cp.Parameter(nonneg=True)
        log_likelihood = cp.sum(
            cp.multiply(y, X @ a) -
            cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ a]).T, axis=0,
                           keepdims=True).T
        )
        problem = cp.Problem(
            cp.Minimize(-log_likelihood + lam * cp.sum_squares(a)))
        X.value = X_np
        lam.value = 1
        # TODO(akshayka): too low but this problem is ill-conditioned
        gradcheck(problem, atol=1e-1, eps=1e-8)
        perturbcheck(problem, atol=1e-4)

    def test_entropy_maximization(self):
        np.random.seed(0)
        n, m, p = 5, 3, 2

        tmp = np.random.rand(n)
        A_np = np.random.randn(m, n)
        b_np = A_np.dot(tmp)
        F_np = np.random.randn(p, n)
        g_np = F_np.dot(tmp) + np.random.rand(p)

        x = cp.Variable(n)
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        F = cp.Parameter((p, n))
        g = cp.Parameter(p)
        obj = cp.Maximize(cp.sum(cp.entr(x)) - cp.sum_squares(x))
        constraints = [A @ x == b,
                       F @ x <= g]
        problem = cp.Problem(obj, constraints)
        A.value = A_np
        b.value = b_np
        F.value = F_np
        g.value = g_np
        gradcheck(problem, atol=1e-2, eps=1e-8)
        perturbcheck(problem, atol=1e-4)

    def test_lml(self):
        np.random.seed(0)
        k = 2
        x = cp.Parameter(4)
        y = cp.Variable(4)
        obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
        cons = [cp.sum(y) == k]
        problem = cp.Problem(cp.Minimize(obj), cons)

        x.value = np.array([1., -1., -1., -1.])
        # TODO(akshayka): This tolerance is too low.
        gradcheck(problem, atol=1e-2)
        perturbcheck(problem, atol=1e-4)

    def test_sdp(self):
        np.random.seed(0)
        n = 3
        p = 3
        C = cp.Parameter((n, n))
        As = [cp.Parameter((n, n)) for _ in range(p)]
        bs = [cp.Parameter((1, 1)) for _ in range(p)]

        C.value = np.random.randn(n, n)
        for A, b in zip(As, bs):
            A.value = np.random.randn(n, n)
            b.value = np.random.randn(1, 1)

        X = cp.Variable((n, n), PSD=True)
        constraints = [cp.trace(As[i] @ X) == bs[i] for i in range(p)]
        problem = cp.Problem(cp.Minimize(cp.trace(C @ X) + cp.sum_squares(X)),
                             constraints)
        gradcheck(problem, atol=1e-3)
        perturbcheck(problem)

    def test_forget_requires_grad(self):
        np.random.seed(0)
        m, n = 20, 5
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(obj))
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        problem.solve()
        with self.assertRaisesRegex(ValueError,
                                    "backward can only be called after calling "
                                    "solve with `requires_grad=True`"):
            problem.backward()
        with self.assertRaisesRegex(ValueError,
                                    "derivative can only be called after calling "
                                    "solve with `requires_grad=True`"):
            problem.derivative()

    def test_infeasible(self):
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        param.value = 1
        try:
            problem.solve(requires_grad=True)
        except cp.SolverError:
            with self.assertRaisesRegex(ValueError, "Backpropagating through "
                                                    "infeasible/unbounded.*"):
                problem.backward()
            with self.assertRaisesRegex(ValueError, "Differentiating through "
                                                    "infeasible/unbounded.*"):
                problem.derivative()

    def test_unbounded(self):
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(x), [x <= param])
        param.value = 1
        try:
            problem.solve(requires_grad=True)
        except cp.SolverError:
            with self.assertRaisesRegex(ValueError, "Backpropagating through "
                                                    "infeasible/unbounded.*"):
                problem.backward()
            with self.assertRaisesRegex(ValueError, "Differentiating through "
                                                    "infeasible/unbounded.*"):
                problem.derivative()

    def test_unsupported_solver(self):
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(x), [x <= param])
        param.value = 1
        with self.assertRaisesRegex(ValueError,
                                    "When requires_grad is True, the "
                                    "only supported solver is SCS.*"):
            problem.solve(cp.ECOS, requires_grad=True)

    def test_zero_in_problem_data(self):
        x = cp.Variable()
        param = cp.Parameter()
        param.value = 0.0
        problem = cp.Problem(cp.Minimize(x), [param * x >= 0])
        data, _, _ = problem.get_problem_data(cp.DIFFCP)
        A = data[s.A]
        self.assertIn(0.0, A.data)
