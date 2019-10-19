import cvxpy as cp
from cvxpy.tests.base_test import BaseTest

import numpy as np


class TestDpp(BaseTest):
    def test_multiply_scalar_params_not_dpp(self):
        x = cp.Parameter()
        product = x * x
        self.assertFalse(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_matmul_params_not_dpp(self):
        X = cp.Parameter((4, 4))
        product = X @ X
        self.assertTrue(product.is_dcp())
        self.assertFalse(product.is_dpp())

    def test_multiply_param_and_variable_is_dpp(self):
        x = cp.Parameter()
        y = cp.Variable()
        product = x * y
        self.assertTrue(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_variable_and_param_is_dpp(self):
        x = cp.Parameter()
        y = cp.Variable()
        product = cp.multiply(y, x)
        self.assertTrue(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_nonlinear_param_and_variable_is_not_dpp(self):
        x = cp.Parameter()
        y = cp.Variable()
        product = cp.exp(x) * y
        self.assertFalse(product.is_dpp())

    def test_multiply_nonlinear_nonneg_param_and_nonneg_variable_is_not_dpp(self):
        x = cp.Parameter(nonneg=True)
        y = cp.Variable(nonneg=True)
        product = cp.exp(x) * y
        self.assertFalse(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_affine_param_and_variable_is_dpp(self):
        x = cp.Parameter()
        y = cp.Variable()
        product = (x + x) * y
        self.assertTrue(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_param_plus_var_times_const(self):
        x = cp.Parameter()
        y = cp.Variable()
        product = (x + y) * 5
        self.assertTrue(product.is_convex())
        self.assertTrue(product.is_dcp())
        self.assertTrue(product.is_dpp())

    def test_multiply_param_and_nonlinear_variable_is_dpp(self):
        x = cp.Parameter(nonneg=True)
        y = cp.Variable()
        product = x * cp.exp(y)
        self.assertTrue(product.is_convex())
        self.assertTrue(product.is_dcp())
        self.assertTrue(product.is_dpp())

    def test_solve_multiply_param_plus_var_times_const(self):
        x = cp.Parameter()
        y = cp.Variable()
        product = (x + y) * 5
        self.assertTrue(product.is_dpp())
        x.value = 2.0
        problem = cp.Problem(cp.Minimize(product), [y == 1])
        value = problem.solve(cp.SCS)
        self.assertAlmostEqual(value, 15)

    def test_paper_example_is_dpp(self):
        F = cp.Parameter((2, 2))
        x = cp.Variable((2, 1))
        g = cp.Parameter((2, 1))
        lambd = cp.Parameter(nonneg=True)
        objective = cp.norm(F @ x - g) + lambd * cp.norm(x)
        constraints = [x >= 0]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        self.assertTrue(objective.is_dpp())
        self.assertTrue(constraints[0].is_dpp())
        self.assertTrue(problem.is_dpp())

    def test_non_dcp_expression_is_not_dpp(self):
        x = cp.Parameter()
        expr = cp.exp(cp.log(x))
        self.assertFalse(expr.is_dpp())

    def test_can_solve_non_dpp_problem(self):
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x * x), [x == y])
        self.assertFalse(problem.is_dpp())
        self.assertTrue(problem.is_dcp())
        self.assertEqual(problem.solve(cp.SCS), 25)
        x.value = 3
        self.assertEqual(problem.solve(cp.SCS), 9)

    def test_solve_dpp_problem(self):
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x + y), [x == y])
        self.assertTrue(problem.is_dpp())
        self.assertTrue(problem.is_dcp())
        self.assertAlmostEqual(problem.solve(cp.SCS), 10)
        x.value = 3
        self.assertAlmostEqual(problem.solve(cp.SCS), 6)

    def test_chain_data_for_non_dpp_problem_evals_params(self):
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x * x), [x == y])
        _, chain, _ = problem.get_problem_data(cp.SCS)
        self.assertFalse(problem.is_dpp())
        self.assertTrue(cp.reductions.eval_params.EvalParams in
                        [type(r) for r in chain.reductions])

    def test_chain_data_for_dpp_problem_does_not_eval_params(self):
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x + y), [x == y])
        _, chain, _ = problem.get_problem_data(cp.SCS)
        self.assertFalse(cp.reductions.eval_params.EvalParams
                         in [type(r) for r in chain.reductions])

    def test_param_quad_form_not_dpp(self):
        x = cp.Variable((2, 1))
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)
        y = cp.quad_form(x, P)
        self.assertFalse(y.is_dpp())
        self.assertTrue(y.is_dcp())

    def test_const_quad_form_is_dpp(self):
        x = cp.Variable((2, 1))
        P = np.eye(2)
        y = cp.quad_form(x, P)
        self.assertTrue(y.is_dpp())
        self.assertTrue(y.is_dcp())

    def test_paper_example_logreg_is_dpp(self):
        N, n = 3, 2
        beta = cp.Variable((n, 1))
        b = cp.Variable((1, 1))
        X = cp.Parameter((N, n))
        Y = np.ones((N, 1))
        lambd1 = cp.Parameter(nonneg=True)
        lambd2 = cp.Parameter(nonneg=True)
        log_likelihood = (1. / N) * cp.sum(
            cp.multiply(Y, X @ beta + b) -
            cp.log_sum_exp(cp.hstack([np.zeros((N, 1)), X @ beta + b]).T,
                           axis=0, keepdims=True).T)
        regularization = -lambd1 * cp.norm(beta, 1) - lambd2 * cp.sum_squares(beta)
        problem = cp.Problem(cp.Maximize(log_likelihood + regularization))
        self.assertTrue(log_likelihood.is_dpp())
        self.assertTrue(problem.is_dcp())
        self.assertTrue(problem.is_dpp())

    def test_paper_example_stoch_control(self):
        n, m = 3, 3
        x = cp.Parameter((n, 1))
        P_sqrt = cp.Parameter((m, m))
        P_21 = cp.Parameter((n, m))
        q = cp.Parameter((m, 1))
        u = cp.Variable((m, 1))
        y = cp.Variable((n, 1))
        objective = 0.5 * cp.sum_squares(P_sqrt @ u) + x.T @ y + q.T @ u
        problem = cp.Problem(cp.Minimize(objective),
                             [cp.norm(u) <= 0.5, y == P_21 @ u])
        self.assertTrue(problem.is_dpp())
        self.assertTrue(problem.is_dcp())

    def test_paper_example_relu(self):
        n = 2
        x = cp.Parameter(n)
        y = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(y - x))
        constraints = [y >= 0]
        problem = cp.Problem(objective, constraints)
        self.assertTrue(problem.is_dpp())
        x.value = np.array([5, 5])
        problem.solve(cp.SCS, eps=1e-8)
        self.assertItemsAlmostEqual(y.value, x.value)
        x.value = np.array([-4, -4])
        problem.solve(cp.SCS, eps=1e-8)
        self.assertItemsAlmostEqual(y.value, np.zeros(2))

    def test_paper_example_opt_net_qp(self):
        m, n = 3, 2
        G = cp.Parameter((m, n))
        h = cp.Parameter((m, 1))
        p = cp.Parameter((n, 1))
        y = cp.Variable((n, 1))
        objective = cp.Minimize(0.5 * cp.sum_squares(y - p))
        constraints = [G @ y <= h]
        problem = cp.Problem(objective, constraints)
        self.assertTrue(problem.is_dpp())

    def test_paper_example_ellipsoidal_constraints(self):
        n = 2
        A_sqrt = cp.Parameter((n, n))
        z = cp.Parameter(n)
        p = cp.Parameter(n)
        y = cp.Variable(n)
        slack = cp.Variable(y.shape)
        objective = cp.Minimize(0.5 * cp.sum_squares(y - p))
        constraints = [0.5 * cp.sum_squares(A_sqrt @ slack) <= 1,
                       slack == y - z]
        problem = cp.Problem(objective, constraints)
        self.assertTrue(problem.is_dpp())
