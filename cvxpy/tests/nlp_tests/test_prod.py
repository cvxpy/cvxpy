import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


class TestProdDNLP:
    """Test that prod expressions are correctly identified as DNLP."""

    def test_prod_is_smooth(self):
        """Test that prod is smooth (both linearizable convex and concave)."""
        x = cp.Variable(3, pos=True)
        p = cp.prod(x)
        assert p.is_atom_smooth()
        assert p.is_smooth()

    def test_prod_is_smooth_convex(self):
        """Test that prod is smooth convex."""
        x = cp.Variable(3, pos=True)
        p = cp.prod(x)
        assert p.is_linearizable_convex()

    def test_prod_is_smooth_concave(self):
        """Test that prod is smooth concave."""
        x = cp.Variable(3, pos=True)
        p = cp.prod(x)
        assert p.is_linearizable_concave()

    def test_prod_composition_smooth(self):
        """Test that compositions with prod are smooth."""
        x = cp.Variable(3, pos=True)
        expr = cp.log(cp.prod(x))
        assert expr.is_smooth()

    def test_prod_problem_is_dnlp(self):
        """Test that a problem with prod is DNLP."""
        x = cp.Variable(3, pos=True)
        obj = cp.Maximize(cp.prod(x))
        constr = [cp.sum(x) <= 3]
        prob = cp.Problem(obj, constr)
        assert prob.is_dnlp()


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestProdIPOPT:
    """Test solving prod problems with IPOPT."""

    def test_prod_maximize_positive(self):
        """Test maximizing prod with positive variables."""
        x = cp.Variable(3, pos=True)
        obj = cp.Maximize(cp.prod(x))
        constr = [cp.sum(x) <= 3]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # Optimal is x = [1, 1, 1] by AM-GM inequality
        assert np.allclose(x.value, [1.0, 1.0, 1.0], atol=1e-4)
        assert np.isclose(prob.value, 1.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_minimize_squared(self):
        """Test minimizing prod squared with mixed sign variables."""
        x = cp.Variable(3)
        x.value = np.array([1.0, -1.0, 2.0])
        obj = cp.Minimize(cp.prod(x)**2)
        constr = [x[0] >= 0.5, x[1] <= -0.5, x[2] >= 1, cp.sum(x) == 2]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # Check constraints are satisfied
        assert x.value[0] >= 0.5 - 1e-4
        assert x.value[1] <= -0.5 + 1e-4
        assert x.value[2] >= 1 - 1e-4
        assert np.isclose(np.sum(x.value), 2.0, atol=1e-4)

        # Check objective
        assert np.isclose(prob.value, np.prod(x.value)**2, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_with_axis(self):
        """Test prod with axis parameter."""
        X = cp.Variable((2, 3), pos=True)
        obj = cp.Maximize(cp.sum(cp.prod(X, axis=1)))
        # Constraint per row so AM-GM applies independently to each row
        constr = [cp.sum(X, axis=1) <= 3]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # By AM-GM, each row should be [1, 1, 1] for max prod
        assert np.allclose(X.value, np.ones((2, 3)), atol=1e-4)
        assert np.isclose(prob.value, 2.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_with_axis_large_matrix(self):
        """Test prod with axis parameter on a larger matrix."""
        X = cp.Variable((4, 5), pos=True)
        obj = cp.Maximize(cp.sum(cp.prod(X, axis=1)))
        # Constraint per row so AM-GM applies independently to each row
        constr = [cp.sum(X, axis=1) <= 5]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # By AM-GM, each row should be [1, 1, 1, 1, 1] for max prod
        assert np.allclose(X.value, np.ones((4, 5)), atol=1e-4)
        assert np.isclose(prob.value, 4.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_with_zero_start(self):
        """Test prod when starting near zero."""
        x = cp.Variable(3)
        x.value = np.array([1.0, 0.1, 2.0])
        obj = cp.Minimize((cp.prod(x) - 1)**2)
        constr = [x[0] >= 0.5, x[2] >= 1, cp.sum(x) == 3]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # Should find a point where prod(x) ≈ 1
        assert np.isclose(np.prod(x.value), 1.0, atol=1e-3)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_negative_values(self):
        """Test prod with negative values in solution."""
        x = cp.Variable(2)
        x.value = np.array([2.0, -2.0])
        obj = cp.Minimize((cp.prod(x) + 4)**2)  # min (x1*x2 + 4)^2
        constr = [x[0] == 2, x[1] == -2]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # With x = [2, -2], prod = -4, so (prod + 4)^2 = 0
        assert np.isclose(x.value[0], 2.0, atol=1e-3)
        assert np.isclose(x.value[1], -2.0, atol=1e-3)
        assert np.isclose(np.prod(x.value), -4.0, atol=1e-3)
        assert np.isclose(prob.value, 0.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_in_constraint(self):
        """Test prod in a constraint."""
        x = cp.Variable(3, pos=True)
        obj = cp.Minimize(cp.sum(x))
        constr = [cp.prod(x) >= 8]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # By AM-GM, min sum when prod = 8 is at x = [2, 2, 2]
        assert np.allclose(x.value, [2.0, 2.0, 2.0], atol=1e-3)
        assert np.isclose(prob.value, 6.0, atol=1e-3)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_log_composition(self):
        """Test log(prod(x)) = sum(log(x)) equivalence."""
        n = 4
        x = cp.Variable(n, pos=True)

        # Using log(prod(x))
        obj1 = cp.Maximize(cp.log(cp.prod(x)))
        constr = [cp.sum(x) <= n]
        prob1 = cp.Problem(obj1, constr)
        prob1.solve(solver=cp.IPOPT, nlp=True, print_level=0)
        val1 = prob1.value
        x1 = x.value.copy()

        checker = DerivativeChecker(prob1)
        checker.run_and_assert()

        # Using sum(log(x))
        obj2 = cp.Maximize(cp.sum(cp.log(x)))
        prob2 = cp.Problem(obj2, constr)
        prob2.solve(solver=cp.IPOPT, nlp=True, print_level=0)
        val2 = prob2.value
        x2 = x.value.copy()

        # Both should give the same result
        assert np.isclose(val1, val2, atol=1e-4)
        assert np.allclose(x1, x2, atol=1e-3)

        checker = DerivativeChecker(prob2)
        checker.run_and_assert()

    def test_prod_exp_log_relationship(self):
        """Test relationship: prod(x) = exp(sum(log(x)))."""
        n = 3
        x = cp.Variable(n, pos=True)

        # Maximize prod(x)
        obj1 = cp.Maximize(cp.prod(x))
        constr = [cp.sum(x) <= n]
        prob1 = cp.Problem(obj1, constr)
        prob1.solve(solver=cp.IPOPT, nlp=True, print_level=0)
        prod_val = prob1.value
        x1 = x.value.copy()

        checker = DerivativeChecker(prob1)
        checker.run_and_assert()

        # Maximize exp(sum(log(x))) which equals prod(x)
        obj2 = cp.Maximize(cp.exp(cp.sum(cp.log(x))))
        prob2 = cp.Problem(obj2, constr)
        prob2.solve(solver=cp.IPOPT, nlp=True, print_level=0)
        exp_sum_log_val = prob2.value
        x2 = x.value.copy()

        # Both should give the same result
        assert np.isclose(prod_val, exp_sum_log_val, atol=1e-4)
        assert np.allclose(x1, x2, atol=1e-3)

        checker = DerivativeChecker(prob2)
        checker.run_and_assert()

    def test_prod_with_axis_zero(self):
        """Test prod with axis=0 parameter on a small matrix."""
        X = cp.Variable((3, 2), pos=True)
        obj = cp.Maximize(cp.sum(cp.prod(X, axis=0)))
        # Constraint per column so AM-GM applies independently to each column
        constr = [cp.sum(X, axis=0) <= 3]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # By AM-GM, each column should be [1, 1, 1] for max prod
        assert np.allclose(X.value, np.ones((3, 2)), atol=1e-4)
        assert np.isclose(prob.value, 2.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_with_axis_zero_large_matrix(self):
        """Test prod with axis=0 parameter on a larger matrix."""
        X = cp.Variable((5, 4), pos=True)
        obj = cp.Maximize(cp.sum(cp.prod(X, axis=0)))
        # Constraint per column so AM-GM applies independently to each column
        constr = [cp.sum(X, axis=0) <= 5]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        # By AM-GM, each column should be [1, 1, 1, 1, 1] for max prod
        assert np.allclose(X.value, np.ones((5, 4)), atol=1e-4)
        assert np.isclose(prob.value, 4.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()

    def test_prod_single_element(self):
        """Test prod of single element."""
        x = cp.Variable(1, pos=True)
        obj = cp.Maximize(cp.prod(x))
        constr = [x <= 5]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.IPOPT, nlp=True, print_level=0)

        assert np.isclose(x.value[0], 5.0, atol=1e-4)
        assert np.isclose(prob.value, 5.0, atol=1e-4)

        checker = DerivativeChecker(prob)
        checker.run_and_assert()