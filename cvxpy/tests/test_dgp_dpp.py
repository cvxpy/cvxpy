"""
Tests for DGP (Disciplined Geometric Programming) with DPP (Disciplined Parametrized Programming).

This verifies the fix for issue #3004: DGP problems can now have get_problem_data(gp=True)
called without all parameters having values.
"""
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
        param.value was None. Now it defers log transformation until solve time.
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
