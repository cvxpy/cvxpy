import warnings

import numpy as np
import pytest

import cvxpy as cp
import cvxpy.error as error
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


class TestDcp(BaseTest):
    def test_multiply_scalar_params_not_dpp(self) -> None:
        x = cp.Parameter()
        product = x * x
        self.assertFalse(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_matmul_params_not_dpp(self) -> None:
        X = cp.Parameter((4, 4))
        product = X @ X
        self.assertTrue(product.is_dcp())
        self.assertFalse(product.is_dpp())

    def test_multiply_param_and_variable_is_dpp(self) -> None:
        x = cp.Parameter()
        y = cp.Variable()
        product = x * y
        self.assertTrue(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_variable_and_param_is_dpp(self) -> None:
        x = cp.Parameter()
        y = cp.Variable()
        product = cp.multiply(y, x)
        self.assertTrue(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_nonlinear_param_and_variable_is_not_dpp(self) -> None:
        x = cp.Parameter()
        y = cp.Variable()
        product = cp.exp(x) * y
        self.assertFalse(product.is_dpp())

    def test_multiply_nonlinear_nonneg_param_and_nonneg_variable_is_not_dpp(self) -> None:
        x = cp.Parameter(nonneg=True)
        y = cp.Variable(nonneg=True)
        product = cp.exp(x) * y
        self.assertFalse(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_affine_param_and_variable_is_dpp(self) -> None:
        x = cp.Parameter()
        y = cp.Variable()
        product = (x + x) * y
        self.assertTrue(product.is_dpp())
        self.assertTrue(product.is_dcp())

    def test_multiply_param_plus_var_times_const(self) -> None:
        x = cp.Parameter()
        y = cp.Variable()
        product = (x + y) * 5
        self.assertTrue(product.is_convex())
        self.assertTrue(product.is_dcp())
        self.assertTrue(product.is_dpp())

    def test_multiply_param_and_nonlinear_variable_is_dpp(self) -> None:
        x = cp.Parameter(nonneg=True)
        y = cp.Variable()
        product = x * cp.exp(y)
        self.assertTrue(product.is_convex())
        self.assertTrue(product.is_dcp())
        self.assertTrue(product.is_dpp())

    def test_nonlinear_equality_not_dpp(self) -> None:
        x = cp.Variable()
        a = cp.Parameter()
        constraint = [x == cp.norm(a)]
        self.assertFalse(constraint[0].is_dcp(dpp=True))
        problem = cp.Problem(cp.Minimize(0), constraint)
        self.assertFalse(problem.is_dcp(dpp=True))

    def test_nonconvex_inequality_not_dpp(self) -> None:
        x = cp.Variable()
        a = cp.Parameter()
        constraint = [x <= cp.norm(a)]
        self.assertFalse(constraint[0].is_dcp(dpp=True))
        problem = cp.Problem(cp.Minimize(0), constraint)
        self.assertFalse(problem.is_dcp(dpp=True))

    def test_solve_multiply_param_plus_var_times_const(self) -> None:
        x = cp.Parameter()
        y = cp.Variable()
        product = (x + y) * 5
        self.assertTrue(product.is_dpp())
        x.value = 2.0
        problem = cp.Problem(cp.Minimize(product), [y == 1])
        value = problem.solve(cp.SCS)
        self.assertAlmostEqual(value, 15)

    def test_paper_example_is_dpp(self) -> None:
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

    def test_non_dcp_expression_is_not_dpp(self) -> None:
        x = cp.Parameter()
        expr = cp.exp(cp.log(x))
        self.assertFalse(expr.is_dpp())

    def test_can_solve_non_dpp_problem(self) -> None:
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x * x), [x == y])
        self.assertFalse(problem.is_dpp())
        self.assertTrue(problem.is_dcp())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(problem.solve(cp.SCS), 25)
        x.value = 3
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertEqual(problem.solve(cp.SCS), 9)

    def test_solve_dpp_problem(self) -> None:
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x + y), [x == y])
        self.assertTrue(problem.is_dpp())
        self.assertTrue(problem.is_dcp())
        self.assertAlmostEqual(problem.solve(cp.SCS), 10)
        x.value = 3
        self.assertAlmostEqual(problem.solve(cp.SCS), 6)

    def test_chain_data_for_non_dpp_problem_evals_params(self) -> None:
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x * x), [x == y])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, chain, _ = problem.get_problem_data(cp.SCS)
        self.assertFalse(problem.is_dpp())
        self.assertTrue(cp.reductions.eval_params.EvalParams in
                        [type(r) for r in chain.reductions])

    def test_chain_data_for_dpp_problem_does_not_eval_params(self) -> None:
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x + y), [x == y])
        _, chain, _ = problem.get_problem_data(cp.SCS)
        self.assertFalse(cp.reductions.eval_params.EvalParams
                         in [type(r) for r in chain.reductions])

    def test_param_quad_form_not_dpp(self) -> None:
        x = cp.Variable((2, 1))
        P = cp.Parameter((2, 2), PSD=True)
        P.value = np.eye(2)
        y = cp.quad_form(x, P)
        self.assertFalse(y.is_dpp())
        self.assertTrue(y.is_dcp())

    def test_const_quad_form_is_dpp(self) -> None:
        x = cp.Variable((2, 1))
        P = np.eye(2)
        y = cp.quad_form(x, P)
        self.assertTrue(y.is_dpp())
        self.assertTrue(y.is_dcp())

    def test_paper_example_logreg_is_dpp(self) -> None:
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

    def test_paper_example_stoch_control(self) -> None:
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

    def test_paper_example_relu(self) -> None:
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

    def test_paper_example_opt_net_qp(self) -> None:
        m, n = 3, 2
        G = cp.Parameter((m, n))
        h = cp.Parameter((m, 1))
        p = cp.Parameter((n, 1))
        y = cp.Variable((n, 1))
        objective = cp.Minimize(0.5 * cp.sum_squares(y - p))
        constraints = [G @ y <= h]
        problem = cp.Problem(objective, constraints)
        self.assertTrue(problem.is_dpp())

    def test_paper_example_ellipsoidal_constraints(self) -> None:
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

    def test_non_dpp_powers(self) -> None:
        s = cp.Parameter(1, nonneg=True)
        x = cp.Variable(1)
        obj = cp.Maximize(x+s)
        cons = [x <= 1]
        prob = cp.Problem(obj, cons)
        s.value = np.array([1.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(solver=cp.SCS, eps=1e-6)
        np.testing.assert_almost_equal(prob.value, 2., decimal=3)

        s = cp.Parameter(1, nonneg=True)
        x = cp.Variable(1)
        obj = cp.Maximize(x+s**2)
        cons = [x <= 1]
        prob = cp.Problem(obj, cons)
        s.value = np.array([1.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(solver=cp.SCS, eps=1e-6)
        np.testing.assert_almost_equal(prob.value, 2., decimal=3)

        s = cp.Parameter(1, nonneg=True)
        x = cp.Variable(1)
        obj = cp.Maximize(cp.multiply(x, s**2))
        cons = [x <= 1]
        prob = cp.Problem(obj, cons)
        s.value = np.array([1.])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            prob.solve(solver=cp.SCS, eps=1e-6)
        np.testing.assert_almost_equal(prob.value, 1., decimal=3)

    def test_ignore_dpp(self) -> None:
        """Test the ignore_dpp flag.
        """
        x = cp.Parameter()
        x.value = 5
        y = cp.Variable()
        problem = cp.Problem(cp.Minimize(x + y), [x == y])
        self.assertTrue(problem.is_dpp())
        self.assertTrue(problem.is_dcp())
        # Basic solve functionality.
        result = problem.solve(cp.SCS, ignore_dpp=True)
        self.assertAlmostEqual(result, 10)

        # enforce_dpp clashes with ignore_dpp
        with pytest.raises(error.DPPError):
            problem.solve(cp.SCS, enforce_dpp=True, ignore_dpp=True)

    def test_quad_over_lin(self) -> None:
        """Test case with parameter in quad_over_lin."""
        # Bug where the second argument to quad_over_lin
        # was a parameter and the problem was solved
        # as a cone program with a quadratic objective:
        # https://github.com/cvxpy/cvxpy/issues/2433
        x = cp.Variable()
        p = cp.Parameter()

        loss = cp.quad_over_lin(x,p) + x
        prob = cp.Problem(
            cp.Minimize(loss),
        )

        with warnings.catch_warnings():
            # TODO(akshayka): Try to emit DPP problems in Dqcp2Dcp
            warnings.filterwarnings('ignore', message=r'.*DPP.*')
            p.value = 1
            prob.solve(solver=cp.CLARABEL)
            sol1 = x.value.copy()
            p.value = 1000
            prob.solve(solver=cp.CLARABEL)
            sol2 = x.value.copy()
            p.value = 1
            prob.solve(solver=cp.CLARABEL)
            sol3 = x.value.copy()
            assert not np.isclose(sol1, sol2)
            assert np.isclose(sol1, sol3)

        # Cannot solve as a QP with DPP.
        with pytest.raises(error.DPPError):
            prob.solve(cp.OSQP, enforce_dpp=True)

        # works for DPP + DGP
        x = cp.Variable(2, pos = True)
        y = cp.Parameter(pos = True, value = 1)
        constraints = [x >= 1]

        objective = cp.quad_over_lin(x,y)

        prob = cp.Problem(cp.Minimize(objective), constraints)
        sol1 = prob.solve(gp=True)
        y.value = 2
        sol2 = prob.solve(gp=True)
        y.value = 1
        sol3 = prob.solve(gp=True)
        assert np.isclose(sol1, 2)
        assert np.isclose(sol2, 1)
        assert np.isclose(sol3, 2)




class TestDgp(BaseTest):
    def test_basic_equality_constraint(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        dgp = cp.Problem(cp.Minimize(x), [x == alpha])

        self.assertTrue(dgp.objective.is_dgp(dpp=True))
        self.assertTrue(dgp.constraints[0].is_dgp(dpp=True))
        self.assertTrue(dgp.is_dgp(dpp=True))
        dgp2dcp = cp.reductions.Dgp2Dcp(dgp)

        dcp = dgp2dcp.reduce()
        self.assertTrue(dcp.is_dpp())

        dgp.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0)

        alpha.value = 2.0
        dgp.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 2.0)

    def test_basic_inequality_constraint(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        constraint = [x + alpha <= x]
        self.assertTrue(constraint[0].is_dgp(dpp=True))
        self.assertTrue(cp.Problem(cp.Minimize(1), constraint).is_dgp(dpp=True))

    def test_nonlla_equality_constraint_not_dpp(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        constraint = [x == x + alpha]
        self.assertFalse(constraint[0].is_dgp(dpp=True))
        self.assertFalse(cp.Problem(cp.Minimize(1), constraint).is_dgp(dpp=True))

    def test_nonllcvx_inequality_constraint_not_dpp(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        constraint = [x <= x + alpha]
        self.assertFalse(constraint[0].is_dgp(dpp=True))
        self.assertFalse(cp.Problem(cp.Minimize(1), constraint).is_dgp(dpp=True))

    def test_param_monomial_is_dpp(self) -> None:
        alpha = cp.Parameter(pos=True)
        beta = cp.Parameter(pos=True)
        kappa = cp.Parameter(pos=True)

        monomial = alpha**1.2 * beta**0.5 * kappa**3 * kappa**2
        self.assertTrue(monomial.is_dgp(dpp=True))

    def test_param_posynomial_is_dpp(self) -> None:
        alpha = cp.Parameter(pos=True)
        beta = cp.Parameter(pos=True)
        kappa = cp.Parameter(pos=True)

        monomial = alpha**1.2 * beta**0.5 * kappa**3 * kappa**2
        posynomial = monomial + alpha**2 * beta**3
        self.assertTrue(posynomial.is_dgp(dpp=True))

    def test_mixed_monomial_is_dpp(self) -> None:
        alpha = cp.Parameter(pos=True)
        beta = cp.Variable(pos=True)
        kappa = cp.Parameter(pos=True)
        tau = cp.Variable(pos=True)

        monomial = alpha**1.2 * beta**0.5 * kappa**3 * kappa**2 * tau
        self.assertTrue(monomial.is_dgp(dpp=True))

    def test_mixed_posynomial_is_dpp(self) -> None:
        alpha = cp.Parameter(pos=True)
        beta = cp.Variable(pos=True)
        kappa = cp.Parameter(pos=True)
        tau = cp.Variable(pos=True)

        monomial = alpha**1.2 * beta**0.5 * kappa**3 * kappa**2 * tau
        posynomial = (monomial + monomial)**3
        self.assertTrue(posynomial.is_dgp(dpp=True))

    def test_nested_power_not_dpp(self) -> None:
        alpha = cp.Parameter(value=1.0)
        x = cp.Variable(pos=True)

        pow1 = x**alpha
        self.assertTrue(pow1.is_dgp(dpp=True))

        pow2 = pow1**alpha
        self.assertFalse(pow2.is_dgp(dpp=True))

    def test_non_dpp_problem_raises_error(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        dgp = cp.Problem(cp.Minimize((alpha*x)**(alpha)), [x == alpha])
        self.assertTrue(dgp.objective.is_dgp())
        self.assertFalse(dgp.objective.is_dgp(dpp=True))

        with self.assertRaises(error.DPPError):
            dgp.solve(solver=cp.SCS, gp=True, enforce_dpp=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dgp.solve(solver=cp.SCS, gp=True, enforce_dpp=False)
            self.assertAlmostEqual(x.value, 1.0)

    def test_basic_monomial(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        beta = cp.Parameter(pos=True, value=2.0)
        x = cp.Variable(pos=True)
        monomial = alpha*beta*x
        problem = cp.Problem(cp.Minimize(monomial), [x == alpha])

        self.assertTrue(problem.is_dgp())
        self.assertTrue(problem.is_dgp(dpp=True))
        self.assertFalse(problem.is_dpp('dcp'))

        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(problem.value, 2.0)

        alpha.value = 3.0
        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 3.0)
        # 3 * 2 * 3 == 18
        self.assertAlmostEqual(problem.value, 18.0)

    def test_basic_posynomial(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        beta = cp.Parameter(pos=True, value=2.0)
        kappa = cp.Parameter(pos=True, value=3.0)
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        monomial_one = alpha*beta*x
        monomial_two = beta*kappa*x*y
        posynomial = monomial_one + monomial_two
        problem = cp.Problem(cp.Minimize(posynomial),
                             [x == alpha, y == beta])

        self.assertTrue(problem.is_dgp())
        self.assertTrue(problem.is_dgp(dpp=True))
        self.assertFalse(problem.is_dpp('dcp'))

        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 2.0)
        # 1*2*1 + 2*3*1*2 == 2 + 12 == 14
        self.assertAlmostEqual(problem.value, 14.0, places=3)

        alpha.value = 4.0
        beta.value = 5.0
        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 4.0)
        self.assertAlmostEqual(y.value, 5.0)
        # 4*5*4 + 5*3*4*5 == 80 + 300 == 380
        self.assertAlmostEqual(problem.value, 380.0, places=3)

    def test_basic_gp(self) -> None:
        x, y, z = cp.Variable((3,), pos=True)
        a = cp.Parameter(pos=True, value=2.0)
        b = cp.Parameter(pos=True, value=1.0)
        constraints = [a*x*y + a*x*z + a*y*z <= b, x >= a*y]
        problem = cp.Problem(cp.Minimize(1/(x*y*z)), constraints)
        self.assertTrue(problem.is_dgp(dpp=True))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(15.59, problem.value, places=2)

    def test_maximum(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        alpha = cp.Parameter(value=0.5)
        beta = cp.Parameter(pos=True, value=3.0)
        kappa = cp.Parameter(pos=True, value=1.0)
        tau = cp.Parameter(pos=True, value=4.0)

        prod1 = x*y**alpha
        prod2 = beta * x*y**alpha
        obj = cp.Minimize(cp.maximum(prod1, prod2))
        constr = [x == kappa, y == tau]

        problem = cp.Problem(obj, constr)
        self.assertTrue(problem.is_dgp(dpp=True))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(1*2, 3*1*2) = 6
        self.assertAlmostEqual(problem.value, 6.0, places=4)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2
        beta.value = 0.5
        kappa.value = 2.0  # x
        tau.value = 3.0    # y
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(2*9, 0.5*2*9) == 18
        self.assertAlmostEqual(problem.value, 18.0, places=4)
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 3.0)

    def test_max(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        alpha = cp.Parameter(value=0.5)
        beta = cp.Parameter(pos=True, value=3.0)
        kappa = cp.Parameter(pos=True, value=1.0)
        tau = cp.Parameter(pos=True, value=4.0)

        prod1 = x*y**alpha
        prod2 = beta * x*y**alpha
        obj = cp.Minimize(cp.max(cp.hstack([prod1, prod2])))
        constr = [x == kappa, y == tau]

        problem = cp.Problem(obj, constr)
        self.assertTrue(problem.is_dgp(dpp=True))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(1*2, 3*1*2) = 6
        self.assertAlmostEqual(problem.value, 6.0, places=4)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2
        beta.value = 0.5
        kappa.value = 2.0  # x
        tau.value = 3.0    # y
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(2*9, 0.5*2*9) == 18
        self.assertAlmostEqual(problem.value, 18.0, places=4)
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 3.0)

    def test_param_in_exponent_and_elsewhere(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0, name='alpha')
        x = cp.Variable(pos=True)
        problem = cp.Problem(cp.Minimize(x**alpha), [x == alpha])

        self.assertTrue(problem.is_dgp(dpp=True))
        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)

        # re-solve (which goes through a separate code path)
        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)

        alpha.value = 3.0
        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 27.0)
        self.assertAlmostEqual(x.value, 3.0)

    def test_minimum(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        alpha = cp.Parameter(pos=True, value=1.0, name='alpha')
        beta = cp.Parameter(pos=True, value=3.0, name='beta')
        prod1 = x * y**alpha
        prod2 = beta * x * y**alpha
        posy = prod1 + prod2
        obj = cp.Maximize(cp.minimum(prod1, prod2, 1/posy))
        constr = [x == alpha, y == 4.0]

        dgp = cp.Problem(obj, constr)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        # prod1 = 1*4, prod2 = 3*4 = 12, 1/posy = 1/(3 +12)
        self.assertAlmostEqual(dgp.value, 1.0 / (4.0 + 12.0))
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2.0
        # prod1 = 2*16, prod2 = 3*2*16 = 96, 1/posy = 1/(32 +96)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(dgp.value, 1.0 / (32.0 + 96.0))
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 4.0)

    def test_min(self) -> None:
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        alpha = cp.Parameter(pos=True, value=1.0, name='alpha')
        beta = cp.Parameter(pos=True, value=3.0, name='beta')
        prod1 = x * y**alpha
        prod2 = beta * x * y**alpha
        posy = prod1 + prod2
        obj = cp.Maximize(cp.min(cp.hstack([prod1, prod2, 1/posy])))
        constr = [x == alpha, y == 4.0]

        dgp = cp.Problem(obj, constr)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        # prod1 = 1*4, prod2 = 3*4 = 12, 1/posy = 1/(3 +12)
        self.assertAlmostEqual(dgp.value, 1.0 / (4.0 + 12.0))
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2.0
        # prod1 = 2*16, prod2 = 3*2*16 = 96, 1/posy = 1/(32 +96)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(dgp.value, 1.0 / (32.0 + 96.0))
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 4.0)

    def test_div(self) -> None:
        alpha = cp.Parameter(pos=True, value=3.0)
        beta = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)

        p = cp.Problem(cp.Minimize(x * y), [y/alpha <= x, y >= beta])
        self.assertAlmostEqual(p.solve(SOLVER, gp=True, enforce_dpp=True),
                               1.0 / 3.0)
        self.assertAlmostEqual(x.value, 1.0 / 3.0)
        self.assertAlmostEqual(y.value, 1.0)

        beta.value = 2.0
        p = cp.Problem(cp.Minimize(x * y), [y/alpha <= x, y >= beta])
        self.assertAlmostEqual(p.solve(SOLVER, gp=True, enforce_dpp=True),
                               4.0 / 3.0)
        self.assertAlmostEqual(x.value, 2.0 / 3.0)
        self.assertAlmostEqual(y.value, 2.0)

    def test_one_minus_pos(self) -> None:
        x = cp.Variable(pos=True)
        obj = cp.Maximize(x)
        alpha = cp.Parameter(pos=True, value=0.1)
        constr = [cp.one_minus_pos(alpha + x) >= 0.4]
        problem = cp.Problem(obj, constr)
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 0.5)
        self.assertAlmostEqual(x.value, 0.5)

        alpha.value = 0.4
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 0.2)
        self.assertAlmostEqual(x.value, 0.2)

    def test_pf_matrix_completion(self) -> None:
        X = cp.Variable((3, 3), pos=True)
        obj = cp.Minimize(cp.pf_eigenvalue(X))
        known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
        known_values = np.array([1.0, 1.9, 0.8, 3.2, 5.9])
        constr = [
          X[known_indices] == known_values,
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
        ]
        problem = cp.Problem(obj, constr)
        # smoke test.
        problem.solve(SOLVER, gp=True)
        optimal_value = problem.value

        param = cp.Parameter(shape=known_values.shape, pos=True,
                             value=0.5*known_values)
        constr = [
          X[known_indices] == param,
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
        ]
        problem = cp.Problem(obj, constr)
        problem.solve(SOLVER, gp=True, enforce_dpp=True)

        # now change param to point to known_value, and check we recover
        # the correct optimal value
        param.value = known_values
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, optimal_value)

    def test_rank_one_nmf(self) -> None:
        X = cp.Variable((3, 3), pos=True)
        x = cp.Variable((3,), pos=True)
        y = cp.Variable((3,), pos=True)
        xy = cp.vstack([x[0] * y, x[1] * y, x[2] * y])
        R = cp.maximum(
          cp.multiply(X, (xy) ** (-1.0)),
          cp.multiply(X ** (-1.0), xy))
        objective = cp.sum(R)
        constraints = [
          X[0, 0] == 1.0,
          X[0, 2] == 1.9,
          X[1, 1] == 0.8,
          X[2, 0] == 3.2,
          X[2, 1] == 5.9,
          x[0] * x[1] * x[2] == 1.0,
        ]
        # smoke test.
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(SOLVER, gp=True)
        optimal_value = prob.value

        param = cp.Parameter(value=-2.0)
        R = cp.maximum(
          cp.multiply(X, (xy) ** (param)),
          cp.multiply(X ** (param), xy))
        objective = cp.sum(R)
        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(SOLVER, gp=True, enforce_dpp=True)

        # now change param to point to known_value, and check we recover the
        # correct optimal value
        param.value = -1.0
        prob.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(prob.value, optimal_value)

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
        # Smoke test.
        problem.solve(SOLVER, gp=True, enforce_dpp=True)

    def test_sum_scalar(self) -> None:
        alpha = cp.Parameter(pos=True, value=1.0)
        w = cp.Variable(pos=True)
        h = cp.Variable(pos=True)
        problem = cp.Problem(cp.Minimize(h),
                             [w*h >= 8, cp.sum(alpha + w) <= 5])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 2)
        self.assertAlmostEqual(h.value, 2)
        self.assertAlmostEqual(w.value, 4)

        alpha.value = 4.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 8)
        self.assertAlmostEqual(h.value, 8)
        self.assertAlmostEqual(w.value, 1)

    def test_sum_vector(self) -> None:
        alpha = cp.Parameter(shape=(2,), pos=True, value=[1.0, 1.0])
        w = cp.Variable(2, pos=True)
        h = cp.Variable(2, pos=True)
        problem = cp.Problem(cp.Minimize(cp.sum(h)),
                             [cp.multiply(w, h) >= 20,
                              cp.sum(alpha + w) <= 10])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 10)
        np.testing.assert_almost_equal(h.value, np.array([5, 5]), decimal=3)
        np.testing.assert_almost_equal(w.value, np.array([4, 4]), decimal=3)

        alpha.value = [4.0, 4.0]
        problem.solve(cp.CLARABEL, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 40, places=3)
        np.testing.assert_almost_equal(h.value, np.array([20, 20]), decimal=3)
        np.testing.assert_almost_equal(w.value, np.array([1, 1]), decimal=3)

    def test_sum_squares_vector(self) -> None:
        alpha = cp.Parameter(shape=(2,), pos=True, value=[1.0, 1.0])
        w = cp.Variable(2, pos=True)
        h = cp.Variable(2, pos=True)
        problem = cp.Problem(cp.Minimize(cp.sum_squares(alpha + h)),
                             [cp.multiply(w, h) >= 20,
                              cp.sum(alpha + w) <= 10])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(w.value, np.array([4, 4]), decimal=3)
        np.testing.assert_almost_equal(h.value, np.array([5, 5]), decimal=3)
        self.assertAlmostEqual(problem.value, 6**2 + 6**2)

        alpha.value = [4.0, 4.0]
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(w.value, np.array([1, 1]), decimal=3)
        np.testing.assert_almost_equal(h.value, np.array([20, 20]), decimal=3)
        np.testing.assert_almost_equal(problem.value, 24**2 + 24**2, decimal=3)

    def test_sum_matrix(self) -> None:
        w = cp.Variable((2, 2), pos=True)
        h = cp.Variable((2, 2), pos=True)
        alpha = cp.Parameter(pos=True, value=1.0)
        problem = cp.Problem(cp.Minimize(alpha*cp.sum(h)),
                             [cp.multiply(w, h) >= 10,
                              cp.sum(w) <= 20])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 8, decimal=4)
        np.testing.assert_almost_equal(h.value, np.array([[2, 2], [2, 2]]), decimal=4)
        np.testing.assert_almost_equal(w.value, np.array([[5, 5], [5, 5]]), decimal=4)

        alpha.value = 2.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 16, decimal=4)

        w = cp.Variable((2, 2), pos=True)
        h = cp.Parameter((2, 2), pos=True)
        h.value = np.ones((2, 2))
        alpha.value = 1.0
        problem = cp.Problem(cp.Minimize(cp.sum(alpha * h)), [w == h])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 4.0, decimal=4)

        h.value = 2.0 * np.ones((2, 2))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 8.0, decimal=4)

        h.value = 3.0 * np.ones((2, 2))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 12.0, decimal=4)

    def test_exp(self) -> None:
        x = cp.Variable(4, pos=True)
        c = cp.Parameter(4, pos=True)
        expr = cp.exp(cp.multiply(c, x))
        self.assertTrue(expr.is_dgp(dpp=True))

        expr = cp.exp(c.T @ x)
        self.assertTrue(expr.is_dgp(dpp=True))

    def test_log(self) -> None:
        x = cp.Variable(4, pos=True)
        c = cp.Parameter(4, pos=True)
        expr = cp.log(cp.multiply(c, x))
        self.assertTrue(expr.is_dgp(dpp=True))

        expr = cp.log(c.T @ x)
        self.assertFalse(expr.is_dgp(dpp=True))

    def test_gmatmul(self) -> None:
        x = cp.Variable(2, pos=True)
        A = cp.Parameter(shape=(2, 2))
        A.value = np.array([[-5, 2], [1, -3]])
        b = np.array([3, 2])
        expr = cp.gmatmul(A, x)
        problem = cp.Problem(cp.Minimize(1.0), [expr == b])
        self.assertTrue(problem.is_dgp(dpp=True))
        problem.solve(solver=cp.SCS, gp=True, enforce_dpp=True)
        sltn = np.exp(np.linalg.solve(A.value, np.log(b)))
        self.assertItemsAlmostEqual(x.value, sltn)

        x_par = cp.Parameter(2, pos=True)
        expr = cp.gmatmul(A, x_par)
        self.assertFalse(expr.is_dgp(dpp=True))
        self.assertTrue(expr.is_dgp(dpp=False))


class TestCallbackParam(BaseTest):
    x = cp.Variable()
    p = cp.Parameter()
    q = cp.Parameter()

    def test_callback_param(self) -> None:
        callback_param = cp.CallbackParam(callback=lambda: self.p.value * self.q.value)
        problem = cp.Problem(cp.Minimize(self.x), [self.x >= callback_param])
        assert problem.is_dpp()
        self.p.value = 1.0
        self.q.value = 4.0
        problem.solve()
        self.assertAlmostEqual(self.x.value, 4.0)

        self.p.value = 2.0
        problem.solve()
        self.assertAlmostEqual(self.x.value, 8.0)

        with pytest.raises(NotImplementedError, match="Cannot set the value of a CallbackParam"):
            callback_param.value = 1.0