import warnings

import numpy as np

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.tests.base_test import BaseTest

warnings.filterwarnings("ignore")

SOLVE_METHODS = [s.SCS, s.ECOS]
EPS_NAME = {s.SCS: "eps",
            s.ECOS: "abstol"}


def perturbcheck(problem, gp: bool = False, solve_methods: list = SOLVE_METHODS,
                 delta: float = 1e-5, atol: float = 1e-6, eps: float = 1e-9, **kwargs) -> None:
    """Checks the analytical derivative against a numerical computation."""

    for solver in solve_methods:
        np.random.seed(0)
        eps_opt = {EPS_NAME[solver]: eps}
        if not problem.parameters():
            problem.solve(solver=s.DIFFCP, gp=gp, requires_grad=True, solve_method=solver,
                          **eps_opt, **kwargs)
            problem.derivative()
            for variable in problem.variables():
                np.testing.assert_equal(variable.delta, 0.0)

        # Compute perturbations analytically
        for param in problem.parameters():
            param.delta = delta * np.random.randn(*param.shape)
        problem.solve(solver=s.DIFFCP, gp=gp, requires_grad=True, solve_method=solver,
                      **eps_opt, **kwargs)
        problem.derivative()
        variable_values = [v.value for v in problem.variables()]
        deltas = [v.delta for v in problem.variables()]

        # Compute perturbations numerically
        old_values = {}
        for param in problem.parameters():
            old_values[param] = param.value
            param.value += param.delta
        problem.solve(solver=solver, gp=gp, **eps_opt, **kwargs)
        num_deltas = [
            v.value - old_value for (v, old_value)
            in zip(problem.variables(), variable_values)]

        for analytical, numerical in zip(deltas, num_deltas):
            np.testing.assert_allclose(analytical, numerical, atol=atol)

        for param in problem.parameters():
            param.value = old_values[param]


def gradcheck(problem, gp: bool = False, solve_methods: list = SOLVE_METHODS,
              delta: float = 1e-5, atol: float = 1e-4, eps: float = 1e-9, **kwargs) -> None:
    """Checks the analytical adjoint derivative against a numerical computation."""
    for solver in solve_methods:
        eps_opt = {EPS_NAME[solver]: eps}
        # Default of 15k iterations for SCS.
        if solver == s.SCS and "max_iters" not in kwargs:
            kwargs["max_iters"] = 15_000

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
            problem.solve(solver=solver, gp=gp, **eps_opt, **kwargs)
            left_solns = np.concatenate([x.value.flatten() for x in problem.variables()])

            values[i] = old - 0.5 * delta
            problem.solve(solver=solver, gp=gp, **eps_opt, **kwargs)
            right_solns = np.concatenate([x.value.flatten() for x in problem.variables()])

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
        problem.solve(solver=s.DIFFCP, requires_grad=True, gp=gp, solve_method=solver,
                      **eps_opt, **kwargs)
        problem.backward()

        for param, numgrad in zip(problem.parameters(), numgrads):
            np.testing.assert_allclose(param.gradient, numgrad, atol=atol)

        for x in problem.variables():
            x.gradient = old_gradients[x]


class TestBackward(BaseTest):
    """Test problem.backward() and problem.derivative()."""
    def setUp(self) -> None:
        try:
            import diffcp
            diffcp  # for flake8
        except ModuleNotFoundError:
            self.skipTest("diffcp not installed.")

    def test_scalar_quadratic(self) -> None:
        b = cp.Parameter()
        x = cp.Variable()
        quadratic = cp.square(x - 2 * b)
        problem = cp.Problem(cp.Minimize(quadratic), [x >= 0])
        b.value = 3.
        problem.solve(solver=cp.DIFFCP, requires_grad=True, eps=1e-10)
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

        problem.solve(solver=cp.DIFFCP, requires_grad=True, eps=1e-10)
        b.delta = 1e-3
        problem.derivative()
        self.assertAlmostEqual(x.delta, 2e-3)

    def test_l1_square(self) -> None:
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

    def test_l1_rectangle(self) -> None:
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
        gradcheck(problem, atol=1e-3)
        perturbcheck(problem, atol=1e-3)

    def test_least_squares(self) -> None:
        np.random.seed(0)
        m, n = 20, 5
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(obj))

        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        gradcheck(problem, solve_methods=[s.SCS])
        perturbcheck(problem, solve_methods=[s.SCS])

    def test_logistic_regression(self) -> None:
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
        gradcheck(problem, solve_methods=[s.SCS], atol=1e-1, eps=1e-8)
        perturbcheck(problem, solve_methods=[s.SCS], atol=1e-4)

    def test_entropy_maximization(self) -> None:
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
        gradcheck(problem, solve_methods=[s.SCS], atol=1e-2, eps=1e-8, max_iters=10_000)
        perturbcheck(problem, solve_methods=[s.SCS], atol=1e-4)

    def test_lml(self) -> None:
        np.random.seed(0)
        k = 2
        x = cp.Parameter(4)
        y = cp.Variable(4)
        obj = -x @ y - cp.sum(cp.entr(y)) - cp.sum(cp.entr(1. - y))
        cons = [cp.sum(y) == k]
        problem = cp.Problem(cp.Minimize(obj), cons)

        x.value = np.array([1., -1., -1., -1.])
        # TODO(akshayka): This tolerance is too low.
        gradcheck(problem, solve_methods=[s.SCS], atol=1e-2)
        perturbcheck(problem, solve_methods=[s.SCS], atol=1e-4)

    def test_sdp(self) -> None:
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
        gradcheck(problem, solve_methods=[s.SCS], atol=1e-3, eps=1e-10)
        perturbcheck(problem, solve_methods=[s.SCS])

    def test_forget_requires_grad(self) -> None:
        np.random.seed(0)
        m, n = 20, 5
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        x = cp.Variable(n)
        obj = cp.sum_squares(A @ x - b) + cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(obj))
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        problem.solve(cp.SCS)
        with self.assertRaisesRegex(ValueError,
                                    "backward can only be called after calling "
                                    "solve with `requires_grad=True`"):
            problem.backward()
        with self.assertRaisesRegex(ValueError,
                                    "derivative can only be called after calling "
                                    "solve with `requires_grad=True`"):
            problem.derivative()

    def test_infeasible(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(param), [x >= 1, x <= -1])
        param.value = 1
        problem.solve(solver=cp.DIFFCP, requires_grad=True)
        with self.assertRaisesRegex(cp.SolverError, "Backpropagating through "
                                                    "infeasible/unbounded.*"):
            problem.backward()
        with self.assertRaisesRegex(ValueError, "Differentiating through "
                                                "infeasible/unbounded.*"):
            problem.derivative()

    def test_unbounded(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(x), [x <= param])
        param.value = 1
        problem.solve(solver=cp.DIFFCP, requires_grad=True)
        with self.assertRaisesRegex(cp.error.SolverError, "Backpropagating through "
                                                          "infeasible/unbounded.*"):
            problem.backward()
        with self.assertRaisesRegex(ValueError, "Differentiating through "
                                                "infeasible/unbounded.*"):
            problem.derivative()

    def test_unsupported_solver(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        problem = cp.Problem(cp.Minimize(x), [x <= param])
        param.value = 1
        with self.assertRaisesRegex(ValueError,
                                    "When requires_grad is True, the "
                                    "only supported solver is SCS.*"):
            problem.solve(cp.ECOS, requires_grad=True)

    def test_zero_in_problem_data(self) -> None:
        x = cp.Variable()
        param = cp.Parameter()
        param.value = 0.0
        problem = cp.Problem(cp.Minimize(x), [param * x >= 0])
        data, _, _ = problem.get_problem_data(cp.DIFFCP)
        A = data[s.A]
        self.assertIn(0.0, A.data)


class TestBackwardDgp(BaseTest):
    """Test problem.backward() and problem.derivative()."""
    def setUp(self) -> None:
        try:
            import diffcp
            diffcp  # for flake8
        except ModuleNotFoundError:
            self.skipTest("diffcp not installed.")

    def test_one_minus_analytic(self) -> None:
        # construct a problem with solution
        # x^\star(\alpha) = 1 - \alpha^2, and derivative
        # x^\star'(\alpha) = -2\alpha
        alpha = cp.Parameter(pos=True)
        x = cp.Variable(pos=True)
        objective = cp.Maximize(x)
        constr = [cp.one_minus_pos(x) >= alpha**2]
        problem = cp.Problem(objective, constr)

        alpha.value = 0.4
        alpha.delta = 1e-5
        problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-5)
        self.assertAlmostEqual(x.value, 1 - 0.4**2, places=3)
        problem.backward()
        problem.derivative()
        self.assertAlmostEqual(alpha.gradient, -2*0.4, places=3)
        self.assertAlmostEqual(x.delta, -2*0.4*1e-5, places=3)

        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

        alpha.value = 0.5
        alpha.delta = 1e-5
        problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-5)
        problem.backward()
        problem.derivative()
        self.assertAlmostEqual(x.value, 1 - 0.5**2, places=3)
        self.assertAlmostEqual(alpha.gradient, -2*0.5, places=3)
        self.assertAlmostEqual(x.delta, -2*0.5*1e-5, places=3)

        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

    def test_analytic_param_in_exponent(self) -> None:
        # construct a problem with solution
        # x^\star(\alpha) = 1 - 2^alpha, and derivative
        # x^\star'(\alpha) = -log(2) * 2^\alpha
        base = 2.0
        alpha = cp.Parameter()
        x = cp.Variable(pos=True)
        objective = cp.Maximize(x)
        constr = [cp.one_minus_pos(x) >= cp.Constant(base)**alpha]
        problem = cp.Problem(objective, constr)

        alpha.value = -1.0
        alpha.delta = 1e-5
        problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-6)
        self.assertAlmostEqual(x.value, 1 - base**(-1.0))
        problem.backward()
        problem.derivative()
        self.assertAlmostEqual(alpha.gradient, -np.log(base)*base**(-1.0))
        self.assertAlmostEqual(x.delta, alpha.gradient*1e-5, places=3)

        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

        alpha.value = -1.2
        alpha.delta = 1e-5
        problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-6)
        self.assertAlmostEqual(x.value, 1 - base**(-1.2))
        problem.backward()
        problem.derivative()
        self.assertAlmostEqual(alpha.gradient, -np.log(base)*base**(-1.2))
        self.assertAlmostEqual(x.delta, alpha.gradient*1e-5, places=3)

        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

    def test_param_used_twice(self) -> None:
        # construct a problem with solution
        # x^\star(\alpha) = 1 - \alpha^2 - alpha^3, and derivative
        # x^\star'(\alpha) = -2\alpha - 3\alpha^2
        alpha = cp.Parameter(pos=True)
        x = cp.Variable(pos=True)
        objective = cp.Maximize(x)
        constr = [cp.one_minus_pos(x) >= alpha**2 + alpha**3]
        problem = cp.Problem(objective, constr)

        alpha.value = 0.4
        alpha.delta = 1e-5
        problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-6)
        self.assertAlmostEqual(x.value, 1 - 0.4**2 - 0.4**3)
        problem.backward()
        problem.derivative()
        self.assertAlmostEqual(alpha.gradient, -2*0.4 - 3*0.4**2)
        self.assertAlmostEqual(x.delta, alpha.gradient*1e-5)

        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

    def test_param_used_in_exponent_and_elsewhere(self) -> None:
        # construct a problem with solution
        # x^\star(\alpha) = 1 - 0.3^alpha - alpha^2, and derivative
        # x^\star'(\alpha) = -log(0.3) * 0.2^\alpha - 2*alpha
        base = 0.3
        alpha = cp.Parameter(pos=True, value=0.5)
        x = cp.Variable(pos=True)
        objective = cp.Maximize(x)
        constr = [cp.one_minus_pos(x) >= cp.Constant(base)**alpha + alpha**2]
        problem = cp.Problem(objective, constr)

        alpha.delta = 1e-5
        problem.solve(solver=cp.DIFFCP, gp=True, requires_grad=True, eps=1e-5)
        self.assertAlmostEqual(x.value, 1 - base**(0.5) - 0.5**2)
        problem.backward()
        problem.derivative()
        self.assertAlmostEqual(alpha.gradient, -np.log(base)*base**(0.5) - 2*0.5)
        self.assertAlmostEqual(x.delta, alpha.gradient*1e-5, places=3)

    def test_basic_gp(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable(pos=True)
        a = cp.Parameter(pos=True)
        b = cp.Parameter(pos=True)
        c = cp.Parameter()
        constraints = [a*(x*y + x*z + y*z) <= b, x >= y**c]
        problem = cp.Problem(cp.Minimize(1/(x*y*z)), constraints)
        self.assertTrue(problem.is_dgp(dpp=True))

        a.value = 2.0
        b.value = 1.0
        c.value = 0.5
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

    def test_maximum(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        a = cp.Parameter(value=0.5)
        b = cp.Parameter(pos=True, value=3.0)
        c = cp.Parameter(pos=True, value=1.0)
        d = cp.Parameter(pos=True, value=4.0)

        prod1 = x * y**a
        prod2 = b * x * y**a
        obj = cp.Minimize(cp.maximum(prod1, prod2))
        constr = [x == c, y == d]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True, atol=1e-3)
        perturbcheck(problem, gp=True, atol=1e-3)

    def test_max(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        a = cp.Parameter(value=0.5)
        b = cp.Parameter(pos=True, value=1.5)
        c = cp.Parameter(pos=True, value=3.0)
        d = cp.Parameter(pos=True, value=1.0)

        prod1 = b * x * y**a
        prod2 = c * x * y**b
        obj = cp.Minimize(cp.max(cp.hstack([prod1, prod2])))
        constr = [x == d, y == b]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True)
        perturbcheck(problem, gp=True)

    def test_div(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        a = cp.Parameter(pos=True, value=3)
        b = cp.Parameter(pos=True, value=1)
        problem = cp.Problem(cp.Minimize(x * y), [y/a <= x, y >= b])
        gradcheck(problem, gp=True)
        perturbcheck(problem, gp=True)

    def test_one_minus_pos(self) -> None:
        x = cp.Variable(pos=True)
        a = cp.Parameter(pos=True, value=3)
        b = cp.Parameter(pos=True, value=0.1)
        obj = cp.Maximize(x)
        constr = [cp.one_minus_pos(a*x) >= a*b]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)

    def test_paper_example_one_minus_pos(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        a = cp.Parameter(pos=True, value=2)
        b = cp.Parameter(pos=True, value=1)
        c = cp.Parameter(pos=True, value=3)
        obj = cp.Minimize(x * y)
        constr = [(y * cp.one_minus_pos(x / y)) ** a >= b, x >= y/c]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-3)
        perturbcheck(problem, solve_methods=[s.SCS], gp=True, atol=1e-3)

    def test_matrix_constraint(self) -> None:
        X = cp.Variable((2, 2), pos=True)
        a = cp.Parameter(pos=True, value=0.1)
        obj = cp.Minimize(cp.geo_mean(cp.vec(X)))
        constr = [cp.diag(X) == a,
                  cp.hstack([X[0, 1], X[1, 0]]) == 2*a]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True)
        perturbcheck(problem, gp=True)

    def test_paper_example_exp_log(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        a = cp.Parameter(pos=True, value=0.2)
        b = cp.Parameter(pos=True, value=0.3)
        obj = cp.Minimize(x * y)
        constr = [cp.exp(a*y/x) <= cp.log(b*y)]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2, max_iters=10_000)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2, max_iters=5000)

    def test_matrix_completion(self) -> None:
        X = cp.Variable((3, 3), pos=True)
        # TODO(akshayka): pf matrix completion not differentiable ...?
        # I could believe that ... or a bug?
        obj = cp.Minimize(cp.sum(X))
        known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
        known_values = np.array([1.0, 1.9, 0.8, 3.2, 5.9])
        param = cp.Parameter(shape=known_values.shape, pos=True,
                             value=known_values)
        beta = cp.Parameter(pos=True, value=1.0)
        constr = [
          X[known_indices] == param,
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == beta,
        ]
        problem = cp.Problem(obj, constr)
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-4)

    def test_rank_one_nmf(self) -> None:
        X = cp.Variable((3, 3), pos=True)
        x = cp.Variable((3,), pos=True)
        y = cp.Variable((3,), pos=True)
        xy = cp.vstack([x[0] * y, x[1] * y, x[2] * y])
        a = cp.Parameter(value=-1.0)
        b = cp.Parameter(pos=True, shape=(6,),
                         value=np.array([1.0, 1.9, 0.8, 3.2, 5.9, 1.0]))
        R = cp.maximum(
          cp.multiply(X, (xy) ** (a)),
          cp.multiply(X ** (a), xy))
        objective = cp.sum(R)
        constraints = [
          X[0, 0] == b[0],
          X[0, 2] == b[1],
          X[1, 1] == b[2],
          X[2, 0] == b[3],
          X[2, 1] == b[4],
          x[0] * x[1] * x[2] == b[5],
        ]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        # SCS struggles to solves this problem (solved/inaccurate, unless
        # max_iters is very high like 10000)
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2, max_iters=1000)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2, max_iters=1000)

    def test_documentation_prob(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable(pos=True)
        a = cp.Parameter(pos=True, value=4.0)
        b = cp.Parameter(pos=True, value=2.0)
        c = cp.Parameter(pos=True, value=10.0)
        d = cp.Parameter(pos=True, value=1.0)

        objective_fn = x * y * z
        constraints = [
          a * x * y * z + b * x * z <= c, x <= b*y, y <= b*x, z >= d]
        problem = cp.Problem(cp.Maximize(objective_fn), constraints)
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-2)

    def test_sum_squares_vector(self) -> None:
        alpha = cp.Parameter(shape=(2,), pos=True, value=[1.0, 1.0])
        beta = cp.Parameter(pos=True, value=20)
        kappa = cp.Parameter(pos=True, value=10)
        w = cp.Variable(2, pos=True)
        h = cp.Variable(2, pos=True)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(alpha + h)),
                             [cp.multiply(w, h) >= beta,
                              cp.sum(alpha + w) <= kappa])
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-1, max_iters=1000)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-1, max_iters=1000)

    def test_sum_matrix(self) -> None:
        w = cp.Variable((2, 2), pos=True)
        h = cp.Variable((2, 2), pos=True)
        alpha = cp.Parameter(pos=True, value=1.0)
        beta = cp.Parameter(pos=True, value=20)
        kappa = cp.Parameter(pos=True, value=10)
        problem = cp.Problem(cp.Minimize(alpha*cp.sum(h)),
                             [cp.multiply(w, h) >= beta,
                              cp.sum(w) <= kappa])
        gradcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-1)
        perturbcheck(problem, gp=True, solve_methods=[s.SCS], atol=1e-1)
