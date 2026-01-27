"""
Tests for DGP (Disciplined Geometric Programming) with DPP (Disciplined Parametrized Programming).

This verifies the fix for issue #3004: DGP problems can now have get_problem_data(gp=True)
called without all parameters having values. This is useful when the problem is DPP.
"""
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.error import ParameterError
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


class TestDgpDpp(BaseTest):
    """Tests for DGP + DPP integration focusing on get_problem_data core functionality."""

    def test_get_problem_data_without_param_values(self) -> None:
        """get_problem_data(gp=True) works with uninitialized parameters.

        This was the original failure case in issue #3004. Previously, DGP
        canonicalization eagerly evaluated np.log(param.value), failing when
        param.value was None.
        """
        alpha = cp.Parameter(pos=True)  # No value set
        x = cp.Variable(pos=True)
        problem = cp.Problem(cp.Minimize(x), [x >= alpha])

        # Should not raise - canonicalization is pure structural transformation
        data, _, _ = problem.get_problem_data(SOLVER, gp=True)
        assert data is not None

        # Now set value and solve using the cached canonicalization
        alpha.value = 2.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 2.0)

    def test_fast_path_with_changing_params(self) -> None:
        """DGP DPP fast path correctly updates parameters across solves.

        Tests that log-transformed parameter values are properly updated
        in the cached param_prog on each solve.
        """
        alpha = cp.Parameter(pos=True)
        x = cp.Variable(pos=True)
        problem = cp.Problem(cp.Minimize(x), [x >= alpha])

        # First solve
        alpha.value = 1.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0, places=4)

        # Second solve with different parameter value
        alpha.value = 2.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 2.0, places=4)

        # Third solve
        alpha.value = 0.5
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 0.5, places=4)

    def test_solve_without_param_value_raises_error(self) -> None:
        """Attempting to solve without setting parameter values raises ParameterError."""
        alpha = cp.Parameter(pos=True)
        x = cp.Variable(pos=True)
        problem = cp.Problem(cp.Minimize(x), [x >= alpha])

        # get_problem_data should work without values
        problem.get_problem_data(SOLVER, gp=True)

        # But solve should raise if values not set
        with pytest.raises(ParameterError, match="must have.*value.*before solving"):
            problem.solve(SOLVER, gp=True, enforce_dpp=True)

    def test_non_dpp_mode_with_ignore_dpp_flag(self) -> None:
        """DGP problems work correctly with ignore_dpp=True (non-DPP mode).

        This tests backward compatibility: even though the DGP canonicalization
        now supports DPP, forcing non-DPP treatment with ignore_dpp=True should
        still work correctly when parameters have values.
        """
        alpha = cp.Parameter(pos=True, value=1.0)
        x = cp.Variable(pos=True)
        problem = cp.Problem(cp.Minimize(x), [x >= alpha])

        # Problem is DPP-compatible but we force non-DPP treatment
        assert problem.is_dpp('dgp')
        problem.solve(SOLVER, gp=True, ignore_dpp=True)
        self.assertAlmostEqual(x.value, 1.0, places=4)

        # Update parameter and solve again with ignore_dpp
        alpha.value = 3.0
        problem.solve(SOLVER, gp=True, ignore_dpp=True)
        self.assertAlmostEqual(x.value, 3.0, places=4)


class TestDgpVariableBounds(BaseTest):
    """Tests for DGP variable bounds (numeric and parametric)."""

    def test_parametric_bounds(self) -> None:
        lb = cp.Parameter(pos=True, value=2.0)
        ub = cp.Parameter(pos=True, value=5.0)
        x = cp.Variable(pos=True, bounds=[lb, ub])

        cp.Problem(cp.Minimize(x)).solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 2.0, places=4)

        cp.Problem(cp.Maximize(x)).solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 5.0, places=4)

    def test_one_sided_parametric_bounds(self) -> None:
        p = cp.Parameter(pos=True, value=0.5)
        x = cp.Variable(pos=True, bounds=[p, None])
        cp.Problem(cp.Minimize(x), [x <= 10.0]).solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 0.5, places=4)

        q = cp.Parameter(pos=True, value=3.0)
        y = cp.Variable(pos=True, bounds=[None, q])
        cp.Problem(cp.Maximize(y)).solve(SOLVER, gp=True)
        self.assertAlmostEqual(y.value, 3.0, places=4)

    def test_mixed_numeric_and_parametric(self) -> None:
        ub = cp.Parameter(pos=True, value=4.0)
        x = cp.Variable(pos=True, bounds=[0.5, ub])
        cp.Problem(cp.Minimize(x)).solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 0.5, places=4)

    def test_re_solve(self) -> None:
        lb = cp.Parameter(pos=True, value=2.0)
        ub = cp.Parameter(pos=True, value=10.0)
        x = cp.Variable(pos=True, bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))

        prob.solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 2.0, places=4)

        lb.value = 3.0
        prob.solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 3.0, places=4)

    def test_is_dpp(self) -> None:
        p = cp.Parameter(pos=True)
        x = cp.Variable(pos=True, bounds=[p, None])
        assert x.is_dpp('dgp')
        assert cp.Problem(cp.Minimize(x), [x <= 10.0]).is_dpp('dgp')

        # Product of params: DPP in DGP (monomial), not in DCP.
        p2 = cp.Parameter(pos=True)
        y = cp.Variable(pos=True, bounds=[p * p2, None])
        assert y.is_dpp('dgp')
        assert not y.is_dpp('dcp')

    def test_enforce_dpp(self) -> None:
        lb = cp.Parameter(pos=True, value=2.0)
        x = cp.Variable(pos=True, bounds=[lb, None])
        prob = cp.Problem(cp.Minimize(x), [x <= 100.0])

        prob.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 2.0, places=4)

        lb.value = 7.0
        prob.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 7.0, places=4)

    def test_get_problem_data_without_param_values(self) -> None:
        p = cp.Parameter(pos=True)
        x = cp.Variable(pos=True, bounds=[p, None])
        data, _, _ = cp.Problem(cp.Minimize(x), [x <= 10.0]).get_problem_data(
            SOLVER, gp=True)
        assert data is not None

    def test_vector_and_matrix(self) -> None:
        lb = cp.Parameter(pos=True, value=2.0)
        x = cp.Variable(3, pos=True, bounds=[lb, None])
        cp.Problem(cp.Minimize(cp.sum(x)), [x <= 100.0]).solve(SOLVER, gp=True)
        np.testing.assert_allclose(x.value, 2.0 * np.ones(3), atol=1e-4)

        X = cp.Variable((2, 3), pos=True, bounds=[lb, None])
        cp.Problem(cp.Minimize(cp.sum(X)), [X <= 100.0]).solve(SOLVER, gp=True)
        np.testing.assert_allclose(X.value, 2.0 * np.ones((2, 3)), atol=1e-4)

    def test_equal_parametric_bounds(self) -> None:
        p = cp.Parameter(pos=True, value=3.0)
        x = cp.Variable(pos=True, bounds=[p, p])
        cp.Problem(cp.Minimize(x)).solve(SOLVER, gp=True)
        self.assertAlmostEqual(x.value, 3.0, places=4)

    def test_solve_without_param_value_raises(self) -> None:
        p = cp.Parameter(pos=True)
        x = cp.Variable(pos=True, bounds=[p, None])
        with pytest.raises(Exception):
            cp.Problem(cp.Minimize(x), [x <= 10.0]).solve(SOLVER, gp=True)
